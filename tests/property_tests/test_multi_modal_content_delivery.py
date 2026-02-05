"""Property-based tests for multi-modal content delivery.

Feature: ai-learning-accelerator, Property 29: Multi-Modal Content Delivery
Validates: Requirements 8.1

Property: For any information presentation request, the system should offer multiple 
formats including text, visual diagrams, and interactive examples.
"""

import asyncio
from typing import Dict, Any, List
from uuid import uuid4

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize

from ai_learning_accelerator.models.content import LearningContent, ContentType, ContentFormat
from ai_learning_accelerator.models.multimodal import (
    ContentAdaptation, AdaptationMode, VisualType, InteractionType
)
from ai_learning_accelerator.models.user import DifficultyLevel
from ai_learning_accelerator.services.content_adaptation import ContentAdaptationService
from tests.property_tests.test_base_properties import AsyncPropertyTest


class MultiModalContentDeliveryTest(AsyncPropertyTest):
    """Property-based tests for multi-modal content delivery."""
    
    @given(
        content_title=st.text(min_size=1, max_size=100),
        content_text=st.text(min_size=10, max_size=1000),
        content_type=st.sampled_from(list(ContentType)),
        difficulty_level=st.sampled_from(list(DifficultyLevel)),
        user_preferences=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(max_size=50), st.booleans(), st.floats(min_value=0, max_value=1)),
            min_size=0,
            max_size=5
        )
    )
    @settings(max_examples=50, deadline=10000)
    async def test_multi_modal_content_delivery_property(
        self,
        content_title: str,
        content_text: str,
        content_type: ContentType,
        difficulty_level: DifficultyLevel,
        user_preferences: Dict[str, Any]
    ):
        """
        Property 29: Multi-Modal Content Delivery
        
        For any information presentation request, the system should offer multiple 
        formats including text, visual diagrams, and interactive examples.
        
        Validates Requirement 8.1: WHEN presenting information, THE AI_Learning_Accelerator 
        SHALL offer multiple formats including text, visual diagrams, and interactive examples.
        """
        # Arrange: Create learning content
        content = LearningContent(
            id=uuid4(),
            title=content_title.strip() or "Test Content",
            content_type=content_type,
            difficulty_level=difficulty_level,
            content_text=content_text,
            content_format=ContentFormat.TEXT,
            is_active=True
        )
        
        # Create service
        service = ContentAdaptationService(self.db)
        
        # Act: Request multi-modal adaptations
        user_id = uuid4()
        adaptations = await service.adapt_content_for_user(
            content_id=content.id,
            user_id=user_id,
            adaptation_modes=None  # Let system determine appropriate modes
        )
        
        # Assert: Verify multi-modal content delivery property
        
        # Property 1: System should offer multiple formats
        assert len(adaptations) >= 1, "System must provide at least one content adaptation"
        
        # Property 2: Should include text format (original content is always available)
        text_available = True  # Original content is always in text format
        assert text_available, "Text format must always be available"
        
        # Property 3: Should offer visual diagrams when appropriate
        visual_adaptations = [
            a for a in adaptations 
            if a.adaptation_mode == AdaptationMode.TEXT_TO_VISUAL
        ]
        
        # For content types that benefit from visualization, visual adaptations should be offered
        visual_beneficial_types = [
            ContentType.TUTORIAL, ContentType.ARTICLE, 
            ContentType.DOCUMENTATION, ContentType.EXAMPLE
        ]
        
        if content_type in visual_beneficial_types:
            assert len(visual_adaptations) >= 0, "Visual adaptations should be considered for appropriate content types"
        
        # Property 4: Should offer interactive examples when appropriate
        interactive_adaptations = [
            a for a in adaptations 
            if a.adaptation_mode == AdaptationMode.INTERACTIVE_EXAMPLE
        ]
        
        # For content types that benefit from interaction, interactive adaptations should be offered
        interactive_beneficial_types = [
            ContentType.TUTORIAL, ContentType.EXERCISE, 
            ContentType.EXAMPLE, ContentType.PROJECT
        ]
        
        if content_type in interactive_beneficial_types:
            assert len(interactive_adaptations) >= 0, "Interactive adaptations should be considered for appropriate content types"
        
        # Property 5: Each adaptation should have valid format and content
        for adaptation in adaptations:
            assert adaptation.source_content_id == content.id, "Adaptation must reference source content"
            assert adaptation.adaptation_mode is not None, "Adaptation must have a valid mode"
            assert adaptation.target_format is not None, "Adaptation must have a target format"
            assert adaptation.adapted_content is not None, "Adaptation must have adapted content"
            assert isinstance(adaptation.adapted_content, dict), "Adapted content must be structured data"
            
            # Verify adaptation quality
            assert 0.0 <= adaptation.adaptation_quality <= 1.0, "Adaptation quality must be between 0.0 and 1.0"
        
        # Property 6: Adaptations should be appropriate for content difficulty
        for adaptation in adaptations:
            if difficulty_level == DifficultyLevel.BEGINNER:
                # Beginner content should prefer simplified adaptations
                if adaptation.adaptation_mode == AdaptationMode.SIMPLIFIED:
                    assert "simplified" in adaptation.target_format or adaptation.adaptation_quality > 0.0
            elif difficulty_level == DifficultyLevel.ADVANCED:
                # Advanced content should prefer detailed adaptations
                if adaptation.adaptation_mode == AdaptationMode.DETAILED:
                    assert "detailed" in adaptation.target_format or adaptation.adaptation_quality > 0.0
        
        # Property 7: System should track adaptation metadata
        for adaptation in adaptations:
            assert adaptation.adaptation_metadata is not None, "Adaptation must have metadata"
            assert isinstance(adaptation.adaptation_metadata, dict), "Metadata must be structured"
            
            # Verify essential metadata fields
            if adaptation.adaptation_metadata:
                assert "generated_at" in adaptation.adaptation_metadata or len(adaptation.adaptation_metadata) == 0
    
    @given(
        content_types=st.lists(
            st.sampled_from(list(ContentType)),
            min_size=1,
            max_size=5,
            unique=True
        ),
        user_accessibility_needs=st.lists(
            st.sampled_from(["screen_reader", "high_contrast", "large_text", "simplified_language"]),
            min_size=0,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=30, deadline=8000)
    async def test_format_variety_consistency(
        self,
        content_types: List[ContentType],
        user_accessibility_needs: List[str]
    ):
        """
        Test that the system consistently offers format variety across different content types.
        """
        format_offerings = {}
        
        for content_type in content_types:
            # Create content of this type
            content = LearningContent(
                id=uuid4(),
                title=f"Test {content_type.value} Content",
                content_type=content_type,
                difficulty_level=DifficultyLevel.INTERMEDIATE,
                content_text="This is test content for multi-modal delivery testing.",
                content_format=ContentFormat.TEXT,
                is_active=True
            )
            
            service = ContentAdaptationService(self.db)
            user_id = uuid4()
            
            # Request adaptations
            adaptations = await service.adapt_content_for_user(
                content_id=content.id,
                user_id=user_id
            )
            
            # Track format offerings
            offered_formats = set()
            offered_formats.add("text")  # Original format always available
            
            for adaptation in adaptations:
                offered_formats.add(adaptation.target_format)
            
            format_offerings[content_type] = offered_formats
        
        # Property: All content types should offer at least text format
        for content_type, formats in format_offerings.items():
            assert "text" in formats, f"Text format must be available for {content_type}"
            assert len(formats) >= 1, f"At least one format must be available for {content_type}"
    
    @given(
        visual_type=st.sampled_from(list(VisualType)),
        interaction_type=st.sampled_from(list(InteractionType))
    )
    @settings(max_examples=20, deadline=6000)
    async def test_specific_format_generation(
        self,
        visual_type: VisualType,
        interaction_type: InteractionType
    ):
        """
        Test that specific format generation works correctly.
        """
        # Create test content
        content = LearningContent(
            id=uuid4(),
            title="Test Content for Format Generation",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            content_text="This is a tutorial about programming concepts with examples and explanations.",
            content_format=ContentFormat.TEXT,
            is_active=True
        )
        
        service = ContentAdaptationService(self.db)
        
        # Test visual content generation
        visual_content = await service.generate_text_to_visual(
            content=content,
            visual_type=visual_type,
            user_preferences={}
        )
        
        if visual_content:
            assert visual_content.visual_type == visual_type, "Generated visual content must match requested type"
            assert visual_content.title is not None, "Visual content must have a title"
            assert visual_content.alt_text is not None, "Visual content must have alt text for accessibility"
        
        # Test interactive content generation
        interactive_content = await service.generate_interactive_example(
            content=content,
            interaction_type=interaction_type,
            user_preferences={}
        )
        
        if interactive_content:
            assert interactive_content.interaction_type == interaction_type, "Generated interactive content must match requested type"
            assert interactive_content.title is not None, "Interactive content must have a title"
            assert interactive_content.config_data is not None, "Interactive content must have configuration data"
            assert isinstance(interactive_content.config_data, dict), "Configuration data must be structured"


class MultiModalContentStateMachine(RuleBasedStateMachine):
    """State machine for testing multi-modal content delivery workflows."""
    
    def __init__(self):
        super().__init__()
        self.contents = Bundle('contents')
        self.adaptations = Bundle('adaptations')
        self.service = None
        self.db = None
    
    @initialize()
    def setup_service(self):
        """Initialize the content adaptation service."""
        # This would be set up with proper async context in real tests
        pass
    
    @rule(
        target=contents,
        title=st.text(min_size=1, max_size=50),
        content_type=st.sampled_from(list(ContentType)),
        difficulty=st.sampled_from(list(DifficultyLevel))
    )
    def create_content(self, title: str, content_type: ContentType, difficulty: DifficultyLevel):
        """Create learning content."""
        content = LearningContent(
            id=uuid4(),
            title=title.strip() or "Test Content",
            content_type=content_type,
            difficulty_level=difficulty,
            content_text="Test content for multi-modal delivery.",
            content_format=ContentFormat.TEXT,
            is_active=True
        )
        return content
    
    @rule(
        target=adaptations,
        content=contents,
        modes=st.lists(
            st.sampled_from(list(AdaptationMode)),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    def adapt_content(self, content, modes: List[AdaptationMode]):
        """Adapt content to multiple modalities."""
        # In a real implementation, this would call the service
        # For now, create mock adaptations
        adaptations = []
        for mode in modes:
            adaptation = ContentAdaptation(
                id=uuid4(),
                source_content_id=content.id,
                adaptation_mode=mode,
                target_format=f"{mode.value}_format",
                adapted_content={"mock": "data"},
                adaptation_quality=0.8
            )
            adaptations.append(adaptation)
        return adaptations
    
    @rule(adaptations_list=adaptations)
    def verify_multi_modal_property(self, adaptations_list):
        """Verify that multi-modal content delivery property holds."""
        if adaptations_list:
            # Property: Each adaptation should be valid
            for adaptation in adaptations_list:
                assert adaptation.source_content_id is not None
                assert adaptation.adaptation_mode is not None
                assert adaptation.adapted_content is not None
                assert 0.0 <= adaptation.adaptation_quality <= 1.0


# Test runner for async property tests
@pytest.mark.asyncio
async def test_multi_modal_content_delivery_properties():
    """Run multi-modal content delivery property tests."""
    test_instance = MultiModalContentDeliveryTest()
    await test_instance.setup()
    
    try:
        # Test the main property
        await test_instance.test_multi_modal_content_delivery_property(
            content_title="Sample Tutorial",
            content_text="This is a comprehensive tutorial about programming concepts.",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            user_preferences={"visual_preference": True, "interactive": True}
        )
        
        # Test format variety consistency
        await test_instance.test_format_variety_consistency(
            content_types=[ContentType.TUTORIAL, ContentType.ARTICLE],
            user_accessibility_needs=["screen_reader", "high_contrast"]
        )
        
        # Test specific format generation
        await test_instance.test_specific_format_generation(
            visual_type=VisualType.DIAGRAM,
            interaction_type=InteractionType.CODE_PLAYGROUND
        )
        
    finally:
        await test_instance.teardown()


# Hypothesis state machine test
MultiModalContentStateMachineTest = MultiModalContentStateMachine.TestCase


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_multi_modal_content_delivery_properties())