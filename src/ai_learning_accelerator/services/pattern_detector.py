"""Pattern detection service for automation engine."""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..models.automation import (
    UserAction, ActionPattern, AutomationOpportunity, AutomationMetrics,
    UserPreference, ActionType, PatternType, AutomationComplexity,
    AutomationStatus
)
from ..models.user import User, SkillLevel
from ..models.context import WorkspaceSession
from ..database import get_async_db

logger = logging.getLogger(__name__)


class PatternDetector:
    """Service for detecting patterns in user actions and identifying automation opportunities."""
    
    def __init__(self):
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        self.action_sequences: Dict[UUID, List[Dict[str, Any]]] = {}
        self.detection_thresholds = {
            "min_frequency": 3,
            "min_confidence": 0.6,
            "max_sequence_length": 20,
            "time_window_hours": 24,
            "similarity_threshold": 0.8
        }
    
    async def track_user_action(
        self,
        user_id: UUID,
        action_type: ActionType,
        action_name: str,
        workspace_path: Optional[str] = None,
        file_path: Optional[str] = None,
        command: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        session_id: Optional[UUID] = None,
        tool_used: Optional[str] = None
    ) -> UserAction:
        """Track a user action for pattern detection."""
        async with get_async_db() as session:
            # Create user action record
            user_action = UserAction(
                user_id=user_id,
                session_id=session_id,
                action_type=action_type,
                action_name=action_name,
                workspace_path=workspace_path,
                file_path=file_path,
                command=command,
                parameters=parameters or {},
                tool_used=tool_used,
                action_timestamp=datetime.utcnow()
            )
            
            # Generate pattern signature
            user_action.pattern_signature = self._generate_action_signature(user_action)
            
            session.add(user_action)
            await session.commit()
            await session.refresh(user_action)
            
            # Update action sequence for pattern detection
            await self._update_action_sequence(user_id, user_action)
            
            # Trigger pattern detection if enough actions accumulated
            await self._check_for_patterns(user_id)
            
            logger.info(f"Tracked action for user {user_id}: {action_type.value} - {action_name}")
            return user_action
    
    def _generate_action_signature(self, action: UserAction) -> str:
        """Generate a signature for an action to help with pattern matching."""
        signature_data = {
            "type": action.action_type.value,
            "name": action.action_name,
            "file_ext": self._extract_file_extension(action.file_path) if action.file_path else None,
            "command_base": self._extract_command_base(action.command) if action.command else None,
            "tool": action.tool_used
        }
        
        # Create hash of signature data
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
    
    def _extract_file_extension(self, file_path: str) -> Optional[str]:
        """Extract file extension from file path."""
        if not file_path:
            return None
        return file_path.split('.')[-1].lower() if '.' in file_path else None
    
    def _extract_command_base(self, command: str) -> Optional[str]:
        """Extract base command from full command string."""
        if not command:
            return None
        return command.split()[0] if command.split() else None
    
    async def _update_action_sequence(self, user_id: UUID, action: UserAction):
        """Update the action sequence for a user."""
        if user_id not in self.action_sequences:
            self.action_sequences[user_id] = []
        
        action_data = {
            "id": action.id,
            "signature": action.pattern_signature,
            "type": action.action_type.value,
            "name": action.action_name,
            "timestamp": action.action_timestamp,
            "file_path": action.file_path,
            "command": action.command,
            "tool": action.tool_used
        }
        
        self.action_sequences[user_id].append(action_data)
        
        # Keep only recent actions (within time window)
        cutoff_time = datetime.utcnow() - timedelta(hours=self.detection_thresholds["time_window_hours"])
        self.action_sequences[user_id] = [
            a for a in self.action_sequences[user_id]
            if a["timestamp"] > cutoff_time
        ]
        
        # Limit sequence length
        max_length = self.detection_thresholds["max_sequence_length"]
        if len(self.action_sequences[user_id]) > max_length:
            self.action_sequences[user_id] = self.action_sequences[user_id][-max_length:]
    
    async def _check_for_patterns(self, user_id: UUID):
        """Check for patterns in user actions."""
        if user_id not in self.action_sequences:
            return
        
        actions = self.action_sequences[user_id]
        if len(actions) < self.detection_thresholds["min_frequency"]:
            return
        
        # Detect different types of patterns
        await self._detect_sequential_patterns(user_id, actions)
        await self._detect_repetitive_patterns(user_id, actions)
        await self._detect_template_patterns(user_id, actions)
    
    async def _detect_sequential_patterns(self, user_id: UUID, actions: List[Dict[str, Any]]):
        """Detect sequential patterns in user actions."""
        # Look for sequences of actions that repeat
        sequence_counts = defaultdict(int)
        
        for seq_length in range(2, min(6, len(actions) + 1)):  # Check sequences of length 2-5
            for i in range(len(actions) - seq_length + 1):
                sequence = tuple(a["signature"] for a in actions[i:i + seq_length])
                sequence_counts[sequence] += 1
        
        # Find sequences that occur frequently enough
        for sequence, count in sequence_counts.items():
            if count >= self.detection_thresholds["min_frequency"]:
                await self._create_pattern_record(
                    user_id=user_id,
                    pattern_type=PatternType.SEQUENTIAL,
                    sequence=sequence,
                    frequency=count,
                    actions=actions
                )
    
    async def _detect_repetitive_patterns(self, user_id: UUID, actions: List[Dict[str, Any]]):
        """Detect repetitive patterns (same action repeated)."""
        signature_counts = Counter(a["signature"] for a in actions)
        
        for signature, count in signature_counts.items():
            if count >= self.detection_thresholds["min_frequency"]:
                # Find all actions with this signature
                matching_actions = [a for a in actions if a["signature"] == signature]
                
                await self._create_pattern_record(
                    user_id=user_id,
                    pattern_type=PatternType.REPETITIVE,
                    sequence=(signature,),
                    frequency=count,
                    actions=matching_actions
                )
    
    async def _detect_template_patterns(self, user_id: UUID, actions: List[Dict[str, Any]]):
        """Detect template-based patterns (similar actions with variations)."""
        # Group actions by type and look for similar patterns
        type_groups = defaultdict(list)
        for action in actions:
            type_groups[action["type"]].append(action)
        
        for action_type, type_actions in type_groups.items():
            if len(type_actions) >= self.detection_thresholds["min_frequency"]:
                # Look for template patterns within this action type
                await self._analyze_template_patterns(user_id, action_type, type_actions)
    
    async def _analyze_template_patterns(self, user_id: UUID, action_type: str, actions: List[Dict[str, Any]]):
        """Analyze actions for template patterns."""
        # Group by similar file paths or commands
        path_patterns = defaultdict(list)
        command_patterns = defaultdict(list)
        
        for action in actions:
            if action.get("file_path"):
                # Extract path pattern (directory structure)
                path_pattern = self._extract_path_pattern(action["file_path"])
                path_patterns[path_pattern].append(action)
            
            if action.get("command"):
                # Extract command pattern
                command_pattern = self._extract_command_pattern(action["command"])
                command_patterns[command_pattern].append(action)
        
        # Check for patterns with sufficient frequency
        for pattern, pattern_actions in path_patterns.items():
            if len(pattern_actions) >= self.detection_thresholds["min_frequency"]:
                await self._create_pattern_record(
                    user_id=user_id,
                    pattern_type=PatternType.TEMPLATE_BASED,
                    sequence=(f"path_template:{pattern}",),
                    frequency=len(pattern_actions),
                    actions=pattern_actions
                )
        
        for pattern, pattern_actions in command_patterns.items():
            if len(pattern_actions) >= self.detection_thresholds["min_frequency"]:
                await self._create_pattern_record(
                    user_id=user_id,
                    pattern_type=PatternType.TEMPLATE_BASED,
                    sequence=(f"command_template:{pattern}",),
                    frequency=len(pattern_actions),
                    actions=pattern_actions
                )
    
    def _extract_path_pattern(self, file_path: str) -> str:
        """Extract a pattern from a file path."""
        # Replace specific names with placeholders
        import re
        
        # Replace numbers with placeholder
        pattern = re.sub(r'\d+', '{num}', file_path)
        
        # Replace specific file names but keep extensions
        parts = pattern.split('/')
        for i, part in enumerate(parts):
            if '.' in part and not part.startswith('.'):
                name, ext = part.rsplit('.', 1)
                parts[i] = f'{{filename}}.{ext}'
        
        return '/'.join(parts)
    
    def _extract_command_pattern(self, command: str) -> str:
        """Extract a pattern from a command."""
        import re
        
        # Replace file paths with placeholder
        pattern = re.sub(r'/[^\s]+', '{filepath}', command)
        
        # Replace numbers with placeholder
        pattern = re.sub(r'\b\d+\b', '{num}', pattern)
        
        # Replace quoted strings with placeholder
        pattern = re.sub(r'"[^"]*"', '{string}', pattern)
        pattern = re.sub(r"'[^']*'", '{string}', pattern)
        
        return pattern
    
    async def _create_pattern_record(
        self,
        user_id: UUID,
        pattern_type: PatternType,
        sequence: Tuple[str, ...],
        frequency: int,
        actions: List[Dict[str, Any]]
    ):
        """Create a pattern record in the database."""
        async with get_async_db() as session:
            # Generate pattern signature
            pattern_signature = hashlib.md5(
                json.dumps({"type": pattern_type.value, "sequence": sequence}, sort_keys=True).encode()
            ).hexdigest()
            
            # Check if pattern already exists
            existing_query = select(ActionPattern).where(
                and_(
                    ActionPattern.user_id == user_id,
                    ActionPattern.pattern_signature == pattern_signature
                )
            )
            existing_result = await session.execute(existing_query)
            existing_pattern = existing_result.scalar_one_or_none()
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.frequency = frequency
                existing_pattern.last_observed_at = datetime.utcnow()
                existing_pattern.confidence_score = min(1.0, frequency / 10.0)  # Simple confidence calculation
                await session.commit()
                pattern = existing_pattern
            else:
                # Create new pattern
                pattern_name = self._generate_pattern_name(pattern_type, sequence, actions)
                
                pattern = ActionPattern(
                    user_id=user_id,
                    pattern_name=pattern_name,
                    pattern_type=pattern_type,
                    pattern_signature=pattern_signature,
                    action_sequence=list(sequence),
                    frequency=frequency,
                    confidence_score=min(1.0, frequency / 10.0),
                    first_observed_at=datetime.utcnow(),
                    last_observed_at=datetime.utcnow()
                )
                
                session.add(pattern)
                await session.commit()
                await session.refresh(pattern)
            
            # Calculate automation score and create opportunity if promising
            automation_score = await self._calculate_automation_score(pattern, actions)
            pattern.automation_score = automation_score
            
            if automation_score >= 0.6:  # Threshold for creating opportunities
                await self._create_automation_opportunity(pattern, actions)
            
            await session.commit()
            
            logger.info(f"Created/updated pattern for user {user_id}: {pattern.pattern_name} (score: {automation_score:.2f})")
    
    def _generate_pattern_name(self, pattern_type: PatternType, sequence: Tuple[str, ...], actions: List[Dict[str, Any]]) -> str:
        """Generate a human-readable name for a pattern."""
        if pattern_type == PatternType.REPETITIVE:
            action_name = actions[0]["name"] if actions else "Unknown Action"
            return f"Repeated {action_name}"
        
        elif pattern_type == PatternType.SEQUENTIAL:
            if len(sequence) <= 3:
                return f"Sequential workflow ({len(sequence)} steps)"
            else:
                return f"Complex workflow ({len(sequence)} steps)"
        
        elif pattern_type == PatternType.TEMPLATE_BASED:
            if sequence[0].startswith("path_template:"):
                return f"File operation pattern"
            elif sequence[0].startswith("command_template:"):
                return f"Command execution pattern"
            else:
                return f"Template-based pattern"
        
        else:
            return f"{pattern_type.value.title()} pattern"
    
    async def _calculate_automation_score(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> float:
        """Calculate automation potential score for a pattern."""
        score = 0.0
        
        # Frequency factor (more frequent = higher score)
        frequency_score = min(1.0, pattern.frequency / 10.0)
        score += frequency_score * 0.3
        
        # Confidence factor
        score += pattern.confidence_score * 0.2
        
        # Pattern type factor
        type_scores = {
            PatternType.REPETITIVE: 0.9,
            PatternType.SEQUENTIAL: 0.8,
            PatternType.TEMPLATE_BASED: 0.7,
            PatternType.CONDITIONAL: 0.6,
            PatternType.PERIODIC: 0.5,
            PatternType.WORKFLOW: 0.8
        }
        score += type_scores.get(pattern.pattern_type, 0.5) * 0.2
        
        # Complexity factor (simpler = higher score)
        complexity_score = 1.0 - (len(pattern.action_sequence) / 20.0)  # Normalize by max sequence length
        score += max(0.0, complexity_score) * 0.15
        
        # Time savings potential
        if actions:
            avg_duration = sum(a.get("duration", 1000) for a in actions) / len(actions)  # Default 1 second
            time_savings_score = min(1.0, avg_duration / 10000.0)  # Normalize by 10 seconds
            score += time_savings_score * 0.15
        
        return min(1.0, score)
    
    async def _create_automation_opportunity(self, pattern: ActionPattern, actions: List[Dict[str, Any]]):
        """Create an automation opportunity from a detected pattern."""
        async with get_async_db() as session:
            # Check if opportunity already exists for this pattern
            existing_query = select(AutomationOpportunity).where(
                AutomationOpportunity.pattern_id == pattern.id
            )
            existing_result = await session.execute(existing_query)
            existing_opportunity = existing_result.scalar_one_or_none()
            
            if existing_opportunity:
                return existing_opportunity  # Already exists
            
            # Determine complexity
            complexity = self._determine_complexity(pattern, actions)
            
            # Calculate time savings potential
            time_saving_potential = self._calculate_time_savings(pattern, actions)
            
            # Generate description and approach
            description = self._generate_opportunity_description(pattern, actions)
            suggested_approach = self._generate_automation_approach(pattern, actions)
            
            opportunity = AutomationOpportunity(
                user_id=pattern.user_id,
                pattern_id=pattern.id,
                title=f"Automate: {pattern.pattern_name}",
                description=description,
                category=self._categorize_opportunity(pattern, actions),
                automation_score=pattern.automation_score,
                complexity=complexity,
                time_saving_potential=time_saving_potential,
                frequency_per_week=pattern.frequency * 7 / 30,  # Rough estimate
                suggested_approach=suggested_approach,
                required_tools=self._identify_required_tools(pattern, actions),
                risk_level=self._assess_risk_level(pattern, actions),
                status=AutomationStatus.DETECTED,
                priority_score=pattern.automation_score
            )
            
            session.add(opportunity)
            await session.commit()
            await session.refresh(opportunity)
            
            logger.info(f"Created automation opportunity: {opportunity.title} (score: {opportunity.automation_score:.2f})")
            return opportunity
    
    def _determine_complexity(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> AutomationComplexity:
        """Determine the complexity of automating a pattern."""
        sequence_length = len(pattern.action_sequence)
        
        # Check for complex operations
        has_file_operations = any(a.get("file_path") for a in actions)
        has_system_commands = any(a.get("command") for a in actions)
        has_multiple_tools = len(set(a.get("tool") for a in actions if a.get("tool"))) > 1
        
        if sequence_length <= 2 and not has_system_commands:
            return AutomationComplexity.SIMPLE
        elif sequence_length <= 5 and not has_multiple_tools:
            return AutomationComplexity.MODERATE
        elif sequence_length <= 10:
            return AutomationComplexity.COMPLEX
        else:
            return AutomationComplexity.ADVANCED
    
    def _calculate_time_savings(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> int:
        """Calculate potential time savings in minutes."""
        if not actions:
            return 5  # Default estimate
        
        # Estimate time per action (in seconds)
        action_time_estimates = {
            ActionType.FILE_OPERATION: 10,
            ActionType.CODE_EDIT: 30,
            ActionType.COMMAND_EXECUTION: 15,
            ActionType.BUILD_OPERATION: 60,
            ActionType.TEST_EXECUTION: 45,
            ActionType.DEBUG_SESSION: 120,
            ActionType.SEARCH_OPERATION: 20,
            ActionType.NAVIGATION: 5,
            ActionType.REFACTORING: 90,
            ActionType.DOCUMENTATION: 60,
            ActionType.VERSION_CONTROL: 20,
            ActionType.DEPLOYMENT: 180
        }
        
        total_seconds = 0
        for action in actions:
            action_type = ActionType(action["type"])
            total_seconds += action_time_estimates.get(action_type, 15)
        
        # Convert to minutes and add some overhead reduction
        minutes = (total_seconds / 60) * 0.8  # 80% of manual time
        return max(1, int(minutes))
    
    def _generate_opportunity_description(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> str:
        """Generate a description for an automation opportunity."""
        if pattern.pattern_type == PatternType.REPETITIVE:
            action_name = actions[0]["name"] if actions else "action"
            return f"You frequently perform the same {action_name} operation. This could be automated to save time and reduce repetitive work."
        
        elif pattern.pattern_type == PatternType.SEQUENTIAL:
            return f"You consistently perform a sequence of {len(pattern.action_sequence)} actions in the same order. This workflow could be automated."
        
        elif pattern.pattern_type == PatternType.TEMPLATE_BASED:
            return f"You perform similar operations with slight variations. A template-based automation could handle these variations automatically."
        
        else:
            return f"A recurring pattern in your workflow has been detected that could benefit from automation."
    
    def _generate_automation_approach(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> str:
        """Generate a suggested automation approach."""
        approaches = []
        
        if any(a.get("command") for a in actions):
            approaches.append("Create a shell script to execute the command sequence")
        
        if any(a.get("file_path") for a in actions):
            approaches.append("Implement file operation automation using Python scripts")
        
        if pattern.pattern_type == PatternType.TEMPLATE_BASED:
            approaches.append("Use template-based generation with configurable parameters")
        
        if not approaches:
            approaches.append("Create a custom automation script tailored to your workflow")
        
        return "; ".join(approaches)
    
    def _categorize_opportunity(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> str:
        """Categorize the automation opportunity."""
        action_types = set(a["type"] for a in actions)
        
        if ActionType.BUILD_OPERATION.value in action_types:
            return "build_automation"
        elif ActionType.TEST_EXECUTION.value in action_types:
            return "testing_automation"
        elif ActionType.FILE_OPERATION.value in action_types:
            return "file_management"
        elif ActionType.VERSION_CONTROL.value in action_types:
            return "version_control"
        elif ActionType.DEPLOYMENT.value in action_types:
            return "deployment"
        else:
            return "workflow_automation"
    
    def _identify_required_tools(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> List[str]:
        """Identify tools required for automation."""
        tools = set()
        
        for action in actions:
            if action.get("tool"):
                tools.add(action["tool"])
            
            if action.get("command"):
                # Extract command base
                command_base = action["command"].split()[0] if action["command"].split() else ""
                if command_base:
                    tools.add(command_base)
        
        return list(tools)
    
    def _assess_risk_level(self, pattern: ActionPattern, actions: List[Dict[str, Any]]) -> str:
        """Assess the risk level of automating a pattern."""
        risk_factors = 0
        
        # Check for risky operations
        for action in actions:
            if action.get("command"):
                command = action["command"].lower()
                if any(risky in command for risky in ["rm", "delete", "drop", "truncate", "format"]):
                    risk_factors += 2
                elif any(moderate in command for moderate in ["mv", "cp", "chmod", "chown"]):
                    risk_factors += 1
            
            if action.get("file_path") and any(sensitive in action["file_path"].lower() 
                                            for sensitive in ["config", "secret", "key", "password"]):
                risk_factors += 1
        
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 1:
            return "medium"
        else:
            return "low"
    
    async def get_automation_opportunities(
        self,
        user_id: UUID,
        min_score: float = 0.6,
        limit: int = 10
    ) -> List[AutomationOpportunity]:
        """Get automation opportunities for a user."""
        async with get_async_db() as session:
            query = select(AutomationOpportunity).where(
                and_(
                    AutomationOpportunity.user_id == user_id,
                    AutomationOpportunity.automation_score >= min_score,
                    AutomationOpportunity.status.in_([
                        AutomationStatus.DETECTED,
                        AutomationStatus.ANALYZED
                    ])
                )
            ).order_by(
                desc(AutomationOpportunity.priority_score),
                desc(AutomationOpportunity.automation_score)
            ).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_user_patterns(
        self,
        user_id: UUID,
        min_frequency: int = 3,
        limit: int = 20
    ) -> List[ActionPattern]:
        """Get detected patterns for a user."""
        async with get_async_db() as session:
            query = select(ActionPattern).where(
                and_(
                    ActionPattern.user_id == user_id,
                    ActionPattern.frequency >= min_frequency,
                    ActionPattern.is_active == True
                )
            ).order_by(
                desc(ActionPattern.automation_score),
                desc(ActionPattern.frequency)
            ).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_pattern_feedback(
        self,
        pattern_id: UUID,
        is_beneficial: bool,
        user_feedback: Optional[str] = None
    ):
        """Update pattern with user feedback."""
        async with get_async_db() as session:
            query = select(ActionPattern).where(ActionPattern.id == pattern_id)
            result = await session.execute(query)
            pattern = result.scalar_one_or_none()
            
            if pattern:
                pattern.is_beneficial = is_beneficial
                if not is_beneficial:
                    pattern.is_active = False
                
                await session.commit()
                
                logger.info(f"Updated pattern {pattern_id} feedback: beneficial={is_beneficial}")
    
    def _detect_patterns_in_actions(self, user_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect patterns in a list of user actions (synchronous version for testing).
        
        This method is used by property-based tests to validate pattern detection logic.
        Returns a list of detected patterns with their properties.
        """
        if len(user_actions) < self.detection_thresholds["min_frequency"]:
            return []
        
        patterns = []
        
        # Detect repetitive patterns
        signature_counts = Counter(a.get("action_type", "") + ":" + a.get("action_name", "") for a in user_actions)
        
        for signature, count in signature_counts.items():
            if count >= self.detection_thresholds["min_frequency"]:
                matching_actions = [a for a in user_actions 
                                  if (a.get("action_type", "") + ":" + a.get("action_name", "")) == signature]
                
                pattern = {
                    "pattern_type": PatternType.REPETITIVE.value,
                    "pattern_signature": signature,
                    "frequency": count,
                    "confidence_score": min(1.0, count / 10.0),
                    "action_sequence": [signature],
                    "automation_score": min(1.0, count / 10.0),
                    "matching_actions": matching_actions
                }
                patterns.append(pattern)
        
        # Detect sequential patterns
        sequence_counts = defaultdict(int)
        
        for seq_length in range(2, min(6, len(user_actions) + 1)):
            for i in range(len(user_actions) - seq_length + 1):
                sequence = tuple(
                    a.get("action_type", "") + ":" + a.get("action_name", "") 
                    for a in user_actions[i:i + seq_length]
                )
                sequence_counts[sequence] += 1
        
        for sequence, count in sequence_counts.items():
            if count >= self.detection_thresholds["min_frequency"]:
                pattern = {
                    "pattern_type": PatternType.SEQUENTIAL.value,
                    "pattern_signature": "|".join(sequence),
                    "frequency": count,
                    "confidence_score": min(1.0, count / 8.0),
                    "action_sequence": list(sequence),
                    "automation_score": min(1.0, count / 8.0),
                    "matching_actions": user_actions  # Simplified for testing
                }
                patterns.append(pattern)
        
        return patterns
    
    def _generate_automation_opportunities_from_pattern(
        self, 
        pattern: Dict[str, Any], 
        user_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Generate automation opportunities from a detected pattern (for testing).
        
        Returns a list of automation opportunities with their properties.
        """
        opportunities = []
        
        # Determine complexity based on pattern
        complexity = AutomationComplexity.SIMPLE
        if pattern["frequency"] > 10:
            complexity = AutomationComplexity.MODERATE
        if len(pattern["action_sequence"]) > 3:
            complexity = AutomationComplexity.COMPLEX
        
        # Calculate time savings (rough estimate)
        time_saving_potential = max(1, pattern["frequency"] * 2)  # 2 minutes per occurrence
        
        # Calculate priority score
        priority_score = pattern["automation_score"] * pattern["frequency"] / 10.0
        
        opportunity = {
            "title": f"Automate {pattern['pattern_type']} pattern",
            "description": f"Detected {pattern['pattern_type']} pattern with {pattern['frequency']} occurrences",
            "automation_score": pattern["automation_score"],
            "complexity": complexity.value,
            "time_saving_potential": time_saving_potential,
            "frequency_per_week": pattern["frequency"] * 7 / 30,  # Rough estimate
            "priority_score": priority_score,
            "pattern_signature": pattern["pattern_signature"],
            "suggested_approach": "Create automation script for detected pattern"
        }
        
        opportunities.append(opportunity)
        return opportunities


# Global pattern detector instance
pattern_detector = PatternDetector()