"""Multi-input interaction service for processing various input modalities."""

import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload

from ..models.interaction import (
    InteractionSession, UserInteraction, VoiceInteraction, 
    GestureInteraction, InputFusion, InteractionFeedback,
    InputCalibration, InputType, InputModality, GestureType, VoiceCommand
)
from ..models.user import User
from ..logging_config import get_logger

logger = get_logger(__name__)


class InteractionService:
    """Service for handling multi-modal user interactions."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_interaction_session(
        self, 
        user_id: UUID,
        device_info: Optional[Dict[str, Any]] = None,
        accessibility_settings: Optional[Dict[str, Any]] = None
    ) -> InteractionSession:
        """Create a new interaction session for a user."""
        try:
            session = InteractionSession(
                user_id=user_id,
                session_token=str(uuid4()),
                device_info=device_info or {},
                accessibility_settings=accessibility_settings or {},
                preferred_input_types=await self._get_user_preferred_inputs(user_id)
            )
            
            self.db.add(session)
            await self.db.commit()
            await self.db.refresh(session)
            
            logger.info("Interaction session created", 
                       session_id=str(session.id), 
                       user_id=str(user_id))
            
            return session
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating interaction session", 
                        error=str(e), 
                        user_id=str(user_id))
            raise
    
    async def process_voice_input(
        self, 
        session_id: UUID,
        audio_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> VoiceInteraction:
        """Process voice input and extract intent."""
        try:
            # Create base interaction record
            interaction = await self._create_base_interaction(
                session_id=session_id,
                input_type=InputType.VOICE,
                input_data=audio_data,
                context=context
            )
            
            # Process voice-specific data
            voice_processing_result = await self._process_voice_data(audio_data)
            
            # Create voice interaction record
            voice_interaction = VoiceInteraction(
                interaction_id=interaction.id,
                audio_url=audio_data.get("audio_url"),
                audio_duration=audio_data.get("duration", 0.0),
                audio_quality=voice_processing_result.get("audio_quality", 0.0),
                raw_transcript=voice_processing_result.get("raw_transcript"),
                processed_transcript=voice_processing_result.get("processed_transcript"),
                language_detected=voice_processing_result.get("language", "en"),
                command_type=voice_processing_result.get("command_type"),
                command_parameters=voice_processing_result.get("command_parameters", {}),
                natural_language_query=voice_processing_result.get("natural_language_query"),
                speech_recognition_confidence=voice_processing_result.get("speech_confidence", 0.0),
                intent_recognition_confidence=voice_processing_result.get("intent_confidence", 0.0),
                noise_level=voice_processing_result.get("noise_level", 0.0)
            )
            
            self.db.add(voice_interaction)
            
            # Update base interaction with results
            interaction.intent_recognized = voice_processing_result.get("intent")
            interaction.confidence_score = voice_processing_result.get("intent_confidence", 0.0)
            interaction.success = voice_processing_result.get("success", True)
            
            await self.db.commit()
            await self.db.refresh(voice_interaction)
            
            logger.info("Voice input processed", 
                       interaction_id=str(interaction.id),
                       intent=interaction.intent_recognized,
                       confidence=interaction.confidence_score)
            
            return voice_interaction
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error processing voice input", 
                        error=str(e), 
                        session_id=str(session_id))
            raise
    
    async def process_gesture_input(
        self, 
        session_id: UUID,
        gesture_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> GestureInteraction:
        """Process gesture input and recognize gestures."""
        try:
            # Create base interaction record
            interaction = await self._create_base_interaction(
                session_id=session_id,
                input_type=InputType.GESTURE,
                input_data=gesture_data,
                context=context
            )
            
            # Process gesture-specific data
            gesture_processing_result = await self._process_gesture_data(gesture_data)
            
            # Create gesture interaction record
            gesture_interaction = GestureInteraction(
                interaction_id=interaction.id,
                gesture_type=gesture_processing_result.get("gesture_type"),
                gesture_data=gesture_data,
                start_position=gesture_processing_result.get("start_position", {}),
                end_position=gesture_processing_result.get("end_position", {}),
                trajectory=gesture_processing_result.get("trajectory", []),
                velocity=gesture_processing_result.get("velocity"),
                acceleration=gesture_processing_result.get("acceleration"),
                recognition_confidence=gesture_processing_result.get("confidence", 0.0),
                gesture_quality=gesture_processing_result.get("quality", 0.0),
                target_element=gesture_processing_result.get("target_element"),
                gesture_area=gesture_processing_result.get("gesture_area", {}),
                action_triggered=gesture_processing_result.get("action"),
                feedback_provided=gesture_processing_result.get("feedback", {})
            )
            
            self.db.add(gesture_interaction)
            
            # Update base interaction with results
            interaction.intent_recognized = gesture_processing_result.get("intent")
            interaction.confidence_score = gesture_processing_result.get("confidence", 0.0)
            interaction.success = gesture_processing_result.get("success", True)
            
            await self.db.commit()
            await self.db.refresh(gesture_interaction)
            
            logger.info("Gesture input processed", 
                       interaction_id=str(interaction.id),
                       gesture_type=gesture_interaction.gesture_type,
                       confidence=interaction.confidence_score)
            
            return gesture_interaction
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error processing gesture input", 
                        error=str(e), 
                        session_id=str(session_id))
            raise
    
    async def fuse_multimodal_inputs(
        self, 
        session_id: UUID,
        interaction_ids: List[UUID],
        fusion_algorithm: str = "weighted_average"
    ) -> InputFusion:
        """Fuse multiple input modalities into a unified intent."""
        try:
            # Get all interactions to fuse
            interactions = await self._get_interactions_by_ids(interaction_ids)
            
            if not interactions:
                raise ValueError("No interactions found for fusion")
            
            # Perform fusion processing
            fusion_result = await self._perform_input_fusion(
                interactions, 
                fusion_algorithm
            )
            
            # Create fusion record
            input_fusion = InputFusion(
                session_id=session_id,
                input_types=[interaction.input_type.value for interaction in interactions],
                interaction_ids=[str(interaction.id) for interaction in interactions],
                fusion_algorithm=fusion_algorithm,
                fusion_parameters=fusion_result.get("parameters", {}),
                combined_intent=fusion_result.get("combined_intent"),
                combined_confidence=fusion_result.get("combined_confidence", 0.0),
                fusion_quality=fusion_result.get("fusion_quality", 0.0),
                conflicts_detected=fusion_result.get("conflicts", []),
                conflict_resolution=fusion_result.get("conflict_resolution", {}),
                final_action=fusion_result.get("final_action"),
                action_parameters=fusion_result.get("action_parameters", {}),
                user_confirmation_required=fusion_result.get("confirmation_required", False)
            )
            
            self.db.add(input_fusion)
            await self.db.commit()
            await self.db.refresh(input_fusion)
            
            logger.info("Multi-modal input fusion completed", 
                       fusion_id=str(input_fusion.id),
                       combined_intent=input_fusion.combined_intent,
                       confidence=input_fusion.combined_confidence)
            
            return input_fusion
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error fusing multi-modal inputs", 
                        error=str(e), 
                        session_id=str(session_id))
            raise
    
    async def calibrate_user_input(
        self, 
        user_id: UUID,
        input_type: InputType,
        calibration_data: Dict[str, Any]
    ) -> InputCalibration:
        """Calibrate user input for improved recognition."""
        try:
            # Check if calibration already exists
            existing_calibration = await self._get_user_calibration(user_id, input_type)
            
            if existing_calibration:
                # Update existing calibration
                calibration_result = await self._update_calibration(
                    existing_calibration, 
                    calibration_data
                )
                
                await self.db.commit()
                await self.db.refresh(existing_calibration)
                
                logger.info("User input calibration updated", 
                           user_id=str(user_id),
                           input_type=input_type.value)
                
                return existing_calibration
            else:
                # Create new calibration
                calibration_result = await self._process_calibration_data(
                    input_type, 
                    calibration_data
                )
                
                calibration = InputCalibration(
                    user_id=user_id,
                    input_type=input_type,
                    calibration_data=calibration_result.get("calibration_data", {}),
                    voice_profile=calibration_result.get("voice_profile", {}),
                    speech_patterns=calibration_result.get("speech_patterns", {}),
                    accent_adaptation=calibration_result.get("accent_adaptation", {}),
                    gesture_preferences=calibration_result.get("gesture_preferences", {}),
                    gesture_sensitivity=calibration_result.get("gesture_sensitivity", 0.5),
                    custom_gestures=calibration_result.get("custom_gestures", {}),
                    accuracy_improvement=calibration_result.get("accuracy_improvement", 0.0),
                    speed_improvement=calibration_result.get("speed_improvement", 0.0),
                    calibration_sessions=1
                )
                
                self.db.add(calibration)
                await self.db.commit()
                await self.db.refresh(calibration)
                
                logger.info("User input calibration created", 
                           user_id=str(user_id),
                           input_type=input_type.value)
                
                return calibration
                
        except Exception as e:
            await self.db.rollback()
            logger.error("Error calibrating user input", 
                        error=str(e), 
                        user_id=str(user_id),
                        input_type=input_type.value)
            raise
    
    # Private helper methods
    
    async def _create_base_interaction(
        self, 
        session_id: UUID,
        input_type: InputType,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> UserInteraction:
        """Create base interaction record."""
        interaction = UserInteraction(
            session_id=session_id,
            input_type=input_type,
            input_data=input_data,
            content_context=context.get("content", {}) if context else {},
            ui_context=context.get("ui", {}) if context else {},
            learning_context=context.get("learning", {}) if context else {}
        )
        
        self.db.add(interaction)
        await self.db.flush()  # Get ID without committing
        
        return interaction
    
    async def _process_voice_data(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice data and extract speech/intent."""
        # This is a simplified implementation
        # In production, this would use speech recognition APIs and NLP models
        
        result = {
            "audio_quality": 0.8,
            "raw_transcript": "Hello, can you help me with this code?",
            "processed_transcript": "Hello, can you help me with this code?",
            "language": "en",
            "speech_confidence": 0.85,
            "intent_confidence": 0.75,
            "noise_level": 0.2,
            "success": True
        }
        
        # Simple intent recognition based on keywords
        transcript = result["raw_transcript"].lower()
        
        if "help" in transcript or "assist" in transcript:
            result["intent"] = "request_help"
            result["command_type"] = VoiceCommand.HELP
        elif "explain" in transcript:
            result["intent"] = "request_explanation"
            result["command_type"] = VoiceCommand.EXPLAIN
        elif "navigate" in transcript or "go to" in transcript:
            result["intent"] = "navigate"
            result["command_type"] = VoiceCommand.NAVIGATE
        elif "read" in transcript:
            result["intent"] = "read_aloud"
            result["command_type"] = VoiceCommand.READ_ALOUD
        elif "search" in transcript or "find" in transcript:
            result["intent"] = "search"
            result["command_type"] = VoiceCommand.SEARCH
        else:
            result["intent"] = "general_query"
            result["natural_language_query"] = result["processed_transcript"]
        
        return result
    
    async def _process_gesture_data(self, gesture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process gesture data and recognize gestures."""
        # This is a simplified implementation
        # In production, this would use gesture recognition algorithms
        
        result = {
            "confidence": 0.8,
            "quality": 0.75,
            "success": True
        }
        
        # Extract gesture information from data
        start_pos = gesture_data.get("start_position", {})
        end_pos = gesture_data.get("end_position", {})
        
        result["start_position"] = start_pos
        result["end_position"] = end_pos
        
        # Simple gesture recognition based on movement
        if start_pos and end_pos:
            dx = end_pos.get("x", 0) - start_pos.get("x", 0)
            dy = end_pos.get("y", 0) - start_pos.get("y", 0)
            
            # Determine gesture type based on movement
            if abs(dx) > abs(dy):
                if dx > 50:
                    result["gesture_type"] = GestureType.SWIPE_RIGHT
                    result["intent"] = "navigate_next"
                    result["action"] = "next_page"
                elif dx < -50:
                    result["gesture_type"] = GestureType.SWIPE_LEFT
                    result["intent"] = "navigate_previous"
                    result["action"] = "previous_page"
            else:
                if dy > 50:
                    result["gesture_type"] = GestureType.SWIPE_DOWN
                    result["intent"] = "scroll_down"
                    result["action"] = "scroll_down"
                elif dy < -50:
                    result["gesture_type"] = GestureType.SWIPE_UP
                    result["intent"] = "scroll_up"
                    result["action"] = "scroll_up"
        
        # If no significant movement, it's a tap
        if "gesture_type" not in result:
            result["gesture_type"] = GestureType.TAP
            result["intent"] = "select"
            result["action"] = "select_element"
        
        # Calculate velocity and acceleration (simplified)
        duration = gesture_data.get("duration", 1.0)
        if duration > 0:
            distance = ((dx ** 2 + dy ** 2) ** 0.5) if start_pos and end_pos else 0
            result["velocity"] = distance / duration
            result["acceleration"] = result["velocity"] / duration
        
        return result
    
    async def _perform_input_fusion(
        self, 
        interactions: List[UserInteraction],
        fusion_algorithm: str
    ) -> Dict[str, Any]:
        """Perform fusion of multiple input modalities."""
        if len(interactions) == 1:
            # Single input, no fusion needed
            return {
                "combined_intent": interactions[0].intent_recognized,
                "combined_confidence": interactions[0].confidence_score,
                "fusion_quality": 1.0,
                "final_action": interactions[0].action_taken,
                "conflicts": [],
                "conflict_resolution": {}
            }
        
        # Multi-modal fusion
        intents = [i.intent_recognized for i in interactions if i.intent_recognized]
        confidences = [i.confidence_score for i in interactions]
        
        # Check for conflicts
        conflicts = []
        if len(set(intents)) > 1:
            conflicts = [{"type": "intent_mismatch", "intents": intents}]
        
        # Simple weighted average fusion
        if fusion_algorithm == "weighted_average":
            combined_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Use highest confidence intent
            if intents and confidences:
                max_confidence_idx = confidences.index(max(confidences))
                combined_intent = intents[max_confidence_idx]
            else:
                combined_intent = "unknown"
        else:
            # Default to first intent
            combined_intent = intents[0] if intents else "unknown"
            combined_confidence = confidences[0] if confidences else 0.0
        
        return {
            "combined_intent": combined_intent,
            "combined_confidence": combined_confidence,
            "fusion_quality": 0.8,  # Placeholder quality score
            "final_action": f"execute_{combined_intent}",
            "conflicts": conflicts,
            "conflict_resolution": {"method": "highest_confidence"} if conflicts else {},
            "confirmation_required": len(conflicts) > 0,
            "parameters": {"algorithm": fusion_algorithm, "input_count": len(interactions)}
        }
    
    async def _get_interactions_by_ids(self, interaction_ids: List[UUID]) -> List[UserInteraction]:
        """Get interactions by their IDs."""
        try:
            stmt = select(UserInteraction).where(UserInteraction.id.in_(interaction_ids))
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error("Error fetching interactions", error=str(e))
            return []
    
    async def _get_user_preferred_inputs(self, user_id: UUID) -> List[str]:
        """Get user's preferred input types."""
        # This would query user preferences
        # For now, return default preferences
        return ["text", "voice", "gesture"]
    
    async def _get_user_calibration(
        self, 
        user_id: UUID, 
        input_type: InputType
    ) -> Optional[InputCalibration]:
        """Get existing user calibration."""
        try:
            stmt = select(InputCalibration).where(
                and_(
                    InputCalibration.user_id == user_id,
                    InputCalibration.input_type == input_type
                )
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user calibration", error=str(e))
            return None
    
    async def _update_calibration(
        self, 
        calibration: InputCalibration,
        calibration_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update existing calibration with new data."""
        # Update calibration fields
        calibration.calibration_sessions += 1
        calibration.calibration_date = datetime.utcnow()
        
        # Merge new calibration data
        if calibration_data.get("voice_profile"):
            calibration.voice_profile.update(calibration_data["voice_profile"])
        
        if calibration_data.get("gesture_preferences"):
            calibration.gesture_preferences.update(calibration_data["gesture_preferences"])
        
        # Update sensitivity if provided
        if "gesture_sensitivity" in calibration_data:
            calibration.gesture_sensitivity = calibration_data["gesture_sensitivity"]
        
        return {"success": True}
    
    async def _process_calibration_data(
        self, 
        input_type: InputType,
        calibration_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process calibration data for new calibration."""
        result = {
            "calibration_data": calibration_data,
            "accuracy_improvement": 0.1,  # Placeholder improvement
            "speed_improvement": 0.05
        }
        
        if input_type == InputType.VOICE:
            result.update({
                "voice_profile": calibration_data.get("voice_profile", {}),
                "speech_patterns": calibration_data.get("speech_patterns", {}),
                "accent_adaptation": calibration_data.get("accent_adaptation", {})
            })
        elif input_type == InputType.GESTURE:
            result.update({
                "gesture_preferences": calibration_data.get("gesture_preferences", {}),
                "gesture_sensitivity": calibration_data.get("gesture_sensitivity", 0.5),
                "custom_gestures": calibration_data.get("custom_gestures", {})
            })
        
        return result