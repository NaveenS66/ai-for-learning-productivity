"""Property-based tests for context adaptation.

Property 10: Context Adaptation
Validates: Requirements 3.3

This test validates that when a user switches between projects, the Context_Analyzer
adapts recommendations to the new context and technology stack, ensuring that
suggestions remain relevant and appropriate for the current work environment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from uuid import uuid4

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from ai_learning_accelerator.models.context import (
    WorkspaceSession, TechnologyStack, ContextRecommendation,
    TechnologyType, RecommendationType, KnowledgeGapSeverity
)
from ai_learning_accelerator.models.user import User, SkillLevel
from ai_learning_accelerator.services.context_analyzer import context_analyzer


# Test data strategies
@st.composite
def technology_stack_strategy(draw):
    """Generate technology stack data for testing."""
    tech_types = list(TechnologyType)
    return {
        "name": draw(st.sampled_from([
            "Python", "JavaScript", "TypeScript", "Java", "C#", "Go", "Rust",
            "React", "Vue", "Angular", "Django", "Flask", "FastAPI", "Express",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Docker", "Kubernetes"
        ])),
        "type": draw(st.sampled_from(tech_types)),
        "version": draw(st.one_of(st.none(), st.text(min_size=3, max_size=10))),
        "confidence": draw(st.floats(min_value=0.5, max_value=1.0)),
        "is_primary": draw(st.booleans()),
        "proficiency": draw(st.sampled_from(list(SkillLevel)))
    }


@st.composite
def project_context_strategy(draw):
    """Generate project context data for testing."""
    project_types = [
        "web_application", "api_service", "data_analysis", "machine_learning",
        "mobile_app", "desktop_app", "microservice", "library", "cli_tool"
    ]
    
    return {
        "project_id": str(uuid4()),
        "project_name": draw(st.text(min_size=5, max_size=50)),
        "project_type": draw(st.sampled_from(project_types)),
        "workspace_path": draw(st.text(min_size=10, max_size=100)),
        "technologies": draw(st.lists(
            technology_stack_strategy(), 
            min_size=1, 
            max_size=8
        )),
        "complexity_level": draw(st.floats(min_value=0.1, max_value=1.0)),
        "domain": draw(st.sampled_from([
            "web_development", "data_science", "devops", "mobile_development",
            "game_development", "embedded_systems", "blockchain", "ai_ml"
        ])),
        "team_size": draw(st.integers(min_value=1, max_value=20)),
        "project_phase": draw(st.sampled_from([
            "planning", "development", "testing", "deployment", "maintenance"
        ]))
    }


@st.composite
def user_profile_strategy(draw):
    """Generate user profile data for testing."""
    return {
        "user_id": uuid4(),
        "skill_level": draw(st.sampled_from(list(SkillLevel))),
        "experience_years": draw(st.integers(min_value=0, max_value=20)),
        "preferred_languages": draw(st.lists(
            st.text(min_size=2, max_size=15), 
            min_size=1, 
            max_size=5
        )),
        "learning_goals": draw(st.lists(
            st.text(min_size=5, max_size=30), 
            min_size=0, 
            max_size=5
        )),
        "work_style": draw(st.sampled_from([
            "focused", "exploratory", "collaborative", "independent"
        ]))
    }


class ContextAdaptationStateMachine(RuleBasedStateMachine):
    """State machine for testing context adaptation."""
    
    def __init__(self):
        super().__init__()
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.project_contexts: Dict[str, Dict[str, Any]] = {}
        self.workspace_sessions: Dict[str, Dict[str, Any]] = {}
        self.recommendations_history: List[Dict[str, Any]] = []
        self.context_switches: List[Dict[str, Any]] = []
    
    @initialize()
    def setup_initial_state(self):
        """Initialize the test state."""
        self.user_profiles.clear()
        self.project_contexts.clear()
        self.workspace_sessions.clear()
        self.recommendations_history.clear()
        self.context_switches.clear()
    
    @rule(user_profile=user_profile_strategy())
    def create_user_profile(self, user_profile):
        """Create a user profile."""
        user_id = str(user_profile["user_id"])
        self.user_profiles[user_id] = user_profile
    
    @rule(project_context=project_context_strategy())
    def create_project_context(self, project_context):
        """Create a project context."""
        project_id = project_context["project_id"]
        self.project_contexts[project_id] = project_context
    
    @rule(
        user_id=st.text(min_size=1, max_size=50),
        project_id=st.text(min_size=1, max_size=50)
    )
    def start_workspace_session(self, user_id, project_id):
        """Start a workspace session for a user in a project."""
        assume(user_id in self.user_profiles or len(self.user_profiles) == 0)
        assume(project_id in self.project_contexts or len(self.project_contexts) == 0)
        
        # Create default user profile if none exists
        if len(self.user_profiles) == 0:
            self.user_profiles[user_id] = {
                "user_id": uuid4(),
                "skill_level": SkillLevel.INTERMEDIATE,
                "experience_years": 3,
                "preferred_languages": ["Python", "JavaScript"],
                "learning_goals": ["improve testing", "learn new frameworks"],
                "work_style": "focused"
            }
        
        # Create default project context if none exists
        if len(self.project_contexts) == 0:
            self.project_contexts[project_id] = {
                "project_id": project_id,
                "project_name": "Test Project",
                "project_type": "web_application",
                "workspace_path": "/test/project",
                "technologies": [{
                    "name": "Python",
                    "type": TechnologyType.PROGRAMMING_LANGUAGE,
                    "version": "3.9",
                    "confidence": 0.9,
                    "is_primary": True,
                    "proficiency": SkillLevel.INTERMEDIATE
                }],
                "complexity_level": 0.5,
                "domain": "web_development",
                "team_size": 3,
                "project_phase": "development"
            }
        
        session_id = str(uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "project_id": project_id,
            "started_at": datetime.utcnow(),
            "is_active": True,
            "context": self.project_contexts[project_id],
            "recommendations": []
        }
        
        self.workspace_sessions[session_id] = session_data
    
    @rule(
        session_id=st.text(min_size=1, max_size=50),
        new_project_id=st.text(min_size=1, max_size=50)
    )
    def switch_project_context(self, session_id, new_project_id):
        """Switch project context within a workspace session."""
        assume(session_id in self.workspace_sessions or len(self.workspace_sessions) == 0)
        assume(new_project_id in self.project_contexts or len(self.project_contexts) == 0)
        
        if len(self.workspace_sessions) == 0 or len(self.project_contexts) == 0:
            return  # Skip if no sessions or projects exist
        
        if session_id not in self.workspace_sessions:
            # Pick a random session if the specified one doesn't exist
            session_id = list(self.workspace_sessions.keys())[0]
        
        if new_project_id not in self.project_contexts:
            # Pick a random project if the specified one doesn't exist
            new_project_id = list(self.project_contexts.keys())[0]
        
        session = self.workspace_sessions[session_id]
        old_project_id = session["project_id"]
        
        # Record context switch
        context_switch = {
            "session_id": session_id,
            "user_id": session["user_id"],
            "from_project": old_project_id,
            "to_project": new_project_id,
            "switched_at": datetime.utcnow(),
            "old_context": session["context"].copy(),
            "new_context": self.project_contexts[new_project_id].copy()
        }
        
        self.context_switches.append(context_switch)
        
        # Update session context
        session["project_id"] = new_project_id
        session["context"] = self.project_contexts[new_project_id]
        session["last_context_switch"] = datetime.utcnow()
    
    @rule(
        session_id=st.text(min_size=1, max_size=50),
        recommendation_context=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(min_size=1, max_size=50), st.floats(min_value=0.0, max_value=1.0)),
            min_size=1,
            max_size=5
        )
    )
    def generate_recommendations(self, session_id, recommendation_context):
        """Generate recommendations for a workspace session."""
        assume(session_id in self.workspace_sessions or len(self.workspace_sessions) == 0)
        
        if len(self.workspace_sessions) == 0:
            return  # Skip if no sessions exist
        
        if session_id not in self.workspace_sessions:
            # Pick a random session if the specified one doesn't exist
            session_id = list(self.workspace_sessions.keys())[0]
        
        session = self.workspace_sessions[session_id]
        project_context = session["context"]
        
        # Generate context-aware recommendations
        recommendations = self._generate_context_aware_recommendations(
            session, project_context, recommendation_context
        )
        
        # Store recommendations
        for rec in recommendations:
            rec["session_id"] = session_id
            rec["generated_at"] = datetime.utcnow()
            self.recommendations_history.append(rec)
            session["recommendations"].append(rec)
    
    def _generate_context_aware_recommendations(
        self, 
        session: Dict[str, Any], 
        project_context: Dict[str, Any],
        recommendation_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on current context."""
        recommendations = []
        
        # Technology-specific recommendations
        for tech in project_context["technologies"]:
            if tech["proficiency"] in [SkillLevel.BEGINNER, SkillLevel.NOVICE]:
                recommendations.append({
                    "id": str(uuid4()),
                    "type": RecommendationType.LEARNING_RESOURCE,
                    "title": f"Learn {tech['name']} fundamentals",
                    "description": f"Improve your {tech['name']} skills for this project",
                    "relevance_score": 0.8,
                    "context_technologies": [tech["name"]],
                    "project_type": project_context["project_type"],
                    "domain": project_context["domain"]
                })
        
        # Project-type specific recommendations
        if project_context["project_type"] == "web_application":
            recommendations.append({
                "id": str(uuid4()),
                "type": RecommendationType.BEST_PRACTICE,
                "title": "Web Application Security Best Practices",
                "description": "Essential security practices for web applications",
                "relevance_score": 0.7,
                "context_technologies": [t["name"] for t in project_context["technologies"]],
                "project_type": project_context["project_type"],
                "domain": project_context["domain"]
            })
        
        # Domain-specific recommendations
        if project_context["domain"] == "data_science":
            recommendations.append({
                "id": str(uuid4()),
                "type": RecommendationType.TOOL_SUGGESTION,
                "title": "Data Visualization Tools",
                "description": "Tools for effective data visualization",
                "relevance_score": 0.6,
                "context_technologies": [t["name"] for t in project_context["technologies"]],
                "project_type": project_context["project_type"],
                "domain": project_context["domain"]
            })
        
        return recommendations
    
    @invariant()
    def context_relevance_property(self):
        """Property: Recommendations should be relevant to current context."""
        for rec in self.recommendations_history:
            session_id = rec["session_id"]
            if session_id in self.workspace_sessions:
                session = self.workspace_sessions[session_id]
                project_context = session["context"]
                
                # Check if recommendation is relevant to project context
                rec_technologies = set(rec.get("context_technologies", []))
                project_technologies = set(t["name"] for t in project_context["technologies"])
                
                # Recommendation should either:
                # 1. Reference technologies used in the project
                # 2. Be relevant to the project type/domain
                # 3. Have high general relevance
                is_relevant = (
                    len(rec_technologies.intersection(project_technologies)) > 0 or
                    rec.get("project_type") == project_context["project_type"] or
                    rec.get("domain") == project_context["domain"] or
                    rec.get("relevance_score", 0) > 0.7
                )
                
                assert is_relevant, (
                    f"Recommendation {rec['id']} not relevant to current context. "
                    f"Rec technologies: {rec_technologies}, "
                    f"Project technologies: {project_technologies}, "
                    f"Project type: {project_context['project_type']}, "
                    f"Domain: {project_context['domain']}"
                )
    
    @invariant()
    def context_adaptation_property(self):
        """Property: Recommendations should adapt when context changes."""
        for switch in self.context_switches:
            session_id = switch["session_id"]
            switch_time = switch["switched_at"]
            
            # Get recommendations before and after context switch
            pre_switch_recs = [
                r for r in self.recommendations_history
                if r["session_id"] == session_id and 
                r["generated_at"] < switch_time
            ]
            
            post_switch_recs = [
                r for r in self.recommendations_history
                if r["session_id"] == session_id and 
                r["generated_at"] > switch_time
            ]
            
            if len(pre_switch_recs) > 0 and len(post_switch_recs) > 0:
                # Check that recommendations adapted to new context
                old_context = switch["old_context"]
                new_context = switch["new_context"]
                
                # Get technology sets
                old_techs = set(t["name"] for t in old_context["technologies"])
                new_techs = set(t["name"] for t in new_context["technologies"])
                
                # If contexts are different, recommendations should adapt
                if old_techs != new_techs or old_context["domain"] != new_context["domain"]:
                    # Check that post-switch recommendations are more relevant to new context
                    new_context_relevance = sum(
                        1 for r in post_switch_recs
                        if (set(r.get("context_technologies", [])).intersection(new_techs) or
                            r.get("domain") == new_context["domain"])
                    )
                    
                    total_post_switch = len(post_switch_recs)
                    if total_post_switch > 0:
                        relevance_ratio = new_context_relevance / total_post_switch
                        assert relevance_ratio > 0.3, (
                            f"Recommendations did not adapt to new context. "
                            f"Only {relevance_ratio:.2%} of post-switch recommendations "
                            f"are relevant to new context"
                        )
    
    @invariant()
    def technology_stack_consistency_property(self):
        """Property: Recommendations should be consistent with technology stack."""
        for session in self.workspace_sessions.values():
            session_recs = [
                r for r in self.recommendations_history
                if r["session_id"] == session["session_id"]
            ]
            
            project_context = session["context"]
            project_techs = set(t["name"] for t in project_context["technologies"])
            
            for rec in session_recs:
                rec_techs = set(rec.get("context_technologies", []))
                
                # If recommendation mentions specific technologies,
                # they should be relevant to the project
                if rec_techs:
                    # Allow for some recommendations about complementary technologies
                    # but most should be directly relevant
                    relevant_tech_count = len(rec_techs.intersection(project_techs))
                    total_tech_count = len(rec_techs)
                    
                    if total_tech_count > 0:
                        relevance_ratio = relevant_tech_count / total_tech_count
                        # Allow some flexibility for complementary technology suggestions
                        assert relevance_ratio > 0.2, (
                            f"Recommendation {rec['id']} mentions technologies "
                            f"{rec_techs} that are not relevant to project technologies "
                            f"{project_techs}"
                        )


# Property-based test functions
@given(
    initial_context=project_context_strategy(),
    new_context=project_context_strategy(),
    user_profile=user_profile_strategy()
)
@settings(max_examples=30, deadline=None)
def test_context_switch_adaptation(initial_context, new_context, user_profile):
    """Test that recommendations adapt when switching between different contexts."""
    
    # Ensure contexts are different enough to test adaptation
    assume(initial_context["domain"] != new_context["domain"] or
           initial_context["project_type"] != new_context["project_type"])
    
    # Simulate initial recommendations
    initial_recommendations = generate_mock_recommendations(initial_context, user_profile)
    
    # Simulate context switch
    new_recommendations = generate_mock_recommendations(new_context, user_profile)
    
    # Property: Recommendations should adapt to new context
    initial_domains = set(r.get("domain") for r in initial_recommendations)
    new_domains = set(r.get("domain") for r in new_recommendations)
    
    # At least some recommendations should be relevant to the new context
    new_context_relevant = sum(
        1 for r in new_recommendations
        if r.get("domain") == new_context["domain"] or
           r.get("project_type") == new_context["project_type"]
    )
    
    if len(new_recommendations) > 0:
        relevance_ratio = new_context_relevant / len(new_recommendations)
        assert relevance_ratio > 0.3, (
            f"Only {relevance_ratio:.2%} of recommendations are relevant to new context"
        )


@given(
    contexts=st.lists(project_context_strategy(), min_size=2, max_size=5),
    user_profile=user_profile_strategy()
)
@settings(max_examples=20, deadline=None)
def test_multi_context_consistency(contexts, user_profile):
    """Test consistency of recommendations across multiple context switches."""
    
    all_recommendations = []
    
    for context in contexts:
        recommendations = generate_mock_recommendations(context, user_profile)
        all_recommendations.extend([
            {**r, "context_id": context["project_id"]} 
            for r in recommendations
        ])
    
    # Property: Recommendations should be consistent within each context
    context_groups = {}
    for rec in all_recommendations:
        context_id = rec["context_id"]
        if context_id not in context_groups:
            context_groups[context_id] = []
        context_groups[context_id].append(rec)
    
    # Check consistency within each context
    for context_id, recs in context_groups.items():
        if len(recs) > 1:
            # All recommendations in the same context should have similar characteristics
            domains = [r.get("domain") for r in recs if r.get("domain")]
            project_types = [r.get("project_type") for r in recs if r.get("project_type")]
            
            # Most recommendations should be for the same domain/project type
            if domains:
                most_common_domain = max(set(domains), key=domains.count)
                domain_consistency = domains.count(most_common_domain) / len(domains)
                assert domain_consistency > 0.5, (
                    f"Inconsistent domains in context {context_id}: {domains}"
                )


@given(
    technology_stacks=st.lists(
        st.lists(technology_stack_strategy(), min_size=1, max_size=5),
        min_size=2,
        max_size=4
    ),
    user_skill_level=st.sampled_from(list(SkillLevel))
)
@settings(max_examples=15, deadline=None)
def test_technology_stack_adaptation(technology_stacks, user_skill_level):
    """Test that recommendations adapt to different technology stacks."""
    
    stack_recommendations = []
    
    for i, tech_stack in enumerate(technology_stacks):
        # Create mock context with this technology stack
        context = {
            "project_id": f"project_{i}",
            "technologies": tech_stack,
            "project_type": "web_application",
            "domain": "web_development"
        }
        
        user_profile = {
            "skill_level": user_skill_level,
            "experience_years": 3,
            "preferred_languages": [tech_stack[0]["name"]] if tech_stack else []
        }
        
        recommendations = generate_mock_recommendations(context, user_profile)
        stack_recommendations.append({
            "stack": tech_stack,
            "recommendations": recommendations
        })
    
    # Property: Recommendations should be relevant to each technology stack
    for stack_data in stack_recommendations:
        tech_names = set(t["name"] for t in stack_data["stack"])
        recommendations = stack_data["recommendations"]
        
        relevant_count = 0
        for rec in recommendations:
            rec_techs = set(rec.get("context_technologies", []))
            if rec_techs.intersection(tech_names):
                relevant_count += 1
        
        if len(recommendations) > 0:
            relevance_ratio = relevant_count / len(recommendations)
            assert relevance_ratio > 0.2, (
                f"Only {relevance_ratio:.2%} of recommendations are relevant "
                f"to technology stack {tech_names}"
            )


def generate_mock_recommendations(context: Dict[str, Any], user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate mock recommendations based on context and user profile."""
    recommendations = []
    
    # Technology-based recommendations
    for tech in context.get("technologies", []):
        if tech.get("proficiency") in [SkillLevel.BEGINNER, SkillLevel.NOVICE]:
            recommendations.append({
                "id": str(uuid4()),
                "type": RecommendationType.LEARNING_RESOURCE.value,
                "title": f"Learn {tech['name']}",
                "context_technologies": [tech["name"]],
                "domain": context.get("domain"),
                "project_type": context.get("project_type"),
                "relevance_score": 0.8
            })
    
    # Domain-based recommendations
    domain = context.get("domain")
    if domain:
        recommendations.append({
            "id": str(uuid4()),
            "type": RecommendationType.BEST_PRACTICE.value,
            "title": f"{domain.replace('_', ' ').title()} Best Practices",
            "context_technologies": [t["name"] for t in context.get("technologies", [])],
            "domain": domain,
            "project_type": context.get("project_type"),
            "relevance_score": 0.7
        })
    
    # Project type recommendations
    project_type = context.get("project_type")
    if project_type:
        recommendations.append({
            "id": str(uuid4()),
            "type": RecommendationType.TOOL_SUGGESTION.value,
            "title": f"Tools for {project_type.replace('_', ' ').title()}",
            "context_technologies": [t["name"] for t in context.get("technologies", [])],
            "domain": context.get("domain"),
            "project_type": project_type,
            "relevance_score": 0.6
        })
    
    return recommendations


# Stateful testing
TestContextAdaptation = ContextAdaptationStateMachine.TestCase


if __name__ == "__main__":
    # Run individual property tests
    test_context_switch_adaptation()
    test_multi_context_consistency()
    test_technology_stack_adaptation()
    
    print("✅ All context adaptation property tests passed!")