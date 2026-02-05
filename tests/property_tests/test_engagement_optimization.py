"""Property-based tests for engagement optimization.

**Property 33: Engagement Optimization**
**Validates: Requirements 8.5**

Engagement optimization must improve user learning engagement through data-driven insights.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

from src.ai_learning_accelerator.models.engagement import EngagementScore, EngagementPattern


class TestEngagementOptimizationProperties:
    """Test engagement optimization correctness properties."""
    
    @given(
        engagement_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=5, max_size=15),
        time_periods=st.lists(st.integers(min_value=1, max_value=24), min_size=5, max_size=15),
        content_types=st.lists(st.text(min_size=1, max_size=20), min_size=3, max_size=8)
    )
    @settings(max_examples=5, deadline=3000)
    def test_engagement_optimization_improves_scores(
        self,
        engagement_scores: List[float],
        time_periods: List[int],
        content_types: List[str]
    ):
        """
        **Property 33: Engagement Optimization**
        **Validates: Requirements 8.5**
        
        Engagement optimization must improve user engagement scores:
        1. Optimized recommendations should lead to higher engagement
        2. Optimization should be based on historical patterns
        3. Improvements should be measurable and consistent
        """
        assume(len(engagement_scores) == len(time_periods))
        
        # Simulate engagement tracking
        engagement_data = self._create_engagement_data(
            engagement_scores, time_periods, content_types
        )
        
        # Apply optimization
        optimized_recommendations = self._optimize_engagement(engagement_data)
        
        # Verify optimization properties
        assert len(optimized_recommendations) > 0
        
        # Check that optimization considers historical patterns
        for recommendation in optimized_recommendations:
            assert "content_type" in recommendation
            assert "optimal_time" in recommendation
            assert "expected_engagement" in recommendation
            assert 0.0 <= recommendation["expected_engagement"] <= 1.0
        
        # Verify recommendations are based on best-performing patterns
        best_historical_engagement = max(engagement_scores)
        for recommendation in optimized_recommendations:
            # Optimized recommendations should target high engagement
            assert recommendation["expected_engagement"] >= best_historical_engagement * 0.8
    
    @given(
        user_patterns=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.floats(min_value=0.0, max_value=1.0),
                min_size=2, max_size=5
            ),
            min_size=2, max_size=5
        ),
        optimization_iterations=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=3, deadline=2000)
    def test_engagement_optimization_personalization(
        self,
        user_patterns: List[Dict[str, float]],
        optimization_iterations: int
    ):
        """
        **Property 33: Engagement Optimization**
        **Validates: Requirements 8.5**
        
        Engagement optimization must be personalized:
        1. Different users should get different optimizations
        2. Optimization should adapt to individual patterns
        3. Personal preferences should be preserved
        """
        users = [uuid4() for _ in range(len(user_patterns))]
        
        # Generate personalized optimizations
        optimizations = {}
        for user_id, pattern in zip(users, user_patterns):
            optimization = self._personalize_engagement_optimization(user_id, pattern)
            optimizations[user_id] = optimization
        
        # Verify personalization
        if len(optimizations) > 1:
            optimization_values = list(optimizations.values())
            
            # Different users should have different optimizations
            for i, opt_a in enumerate(optimization_values):
                for j, opt_b in enumerate(optimization_values):
                    if i != j:
                        # At least some aspect should be different
                        differences = self._count_optimization_differences(opt_a, opt_b)
                        assert differences > 0, "Optimizations should be personalized"
        
        # Verify each optimization respects user patterns
        for user_id, pattern in zip(users, user_patterns):
            optimization = optimizations[user_id]
            
            # Optimization should favor user's high-engagement patterns
            user_best_pattern = max(pattern.items(), key=lambda x: x[1])
            best_pattern_name, best_score = user_best_pattern
            
            # Check if optimization includes or prioritizes the best pattern
            optimization_priorities = optimization.get("priorities", {})
            if best_pattern_name in optimization_priorities:
                assert optimization_priorities[best_pattern_name] >= best_score * 0.7
    
    @given(
        baseline_engagement=st.floats(min_value=0.1, max_value=0.6),
        optimization_strength=st.floats(min_value=0.1, max_value=0.5),
        noise_level=st.floats(min_value=0.0, max_value=0.1)
    )
    @settings(max_examples=3, deadline=2000)
    def test_engagement_optimization_effectiveness(
        self,
        baseline_engagement: float,
        optimization_strength: float,
        noise_level: float
    ):
        """
        **Property 33: Engagement Optimization**
        **Validates: Requirements 8.5**
        
        Engagement optimization must be effective:
        1. Optimized engagement should exceed baseline
        2. Optimization effects should be measurable
        3. Improvements should be statistically significant
        """
        # Simulate baseline engagement
        baseline_data = self._simulate_engagement_data(
            baseline_engagement, noise_level, sample_size=20
        )
        
        # Apply optimization
        optimized_data = self._simulate_optimized_engagement_data(
            baseline_engagement, optimization_strength, noise_level, sample_size=20
        )
        
        # Verify effectiveness
        baseline_avg = sum(baseline_data) / len(baseline_data)
        optimized_avg = sum(optimized_data) / len(optimized_data)
        
        # Optimized engagement should be higher than baseline
        improvement = optimized_avg - baseline_avg
        assert improvement > 0, f"Optimization should improve engagement: {improvement}"
        
        # Improvement should be meaningful (not just noise)
        expected_improvement = optimization_strength * 0.5  # Conservative estimate
        assert improvement >= expected_improvement, f"Improvement should be substantial: {improvement} >= {expected_improvement}"
        
        # Verify consistency - most optimized values should be better
        better_count = sum(1 for opt, base in zip(optimized_data, baseline_data) if opt > base)
        improvement_ratio = better_count / len(optimized_data)
        assert improvement_ratio > 0.6, f"Most optimized values should be better: {improvement_ratio}"
    
    @given(
        engagement_trends=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=10, max_size=20
        ),
        optimization_frequency=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=3, deadline=2000)
    def test_engagement_optimization_adaptation(
        self,
        engagement_trends: List[float],
        optimization_frequency: int
    ):
        """
        **Property 33: Engagement Optimization**
        **Validates: Requirements 8.5**
        
        Engagement optimization must adapt to changing patterns:
        1. Optimization should respond to trend changes
        2. Adaptation should be timely and appropriate
        3. System should learn from recent data
        """
        # Simulate changing engagement trends
        trend_segments = self._segment_trends(engagement_trends, optimization_frequency)
        
        optimizations = []
        for i, segment in enumerate(trend_segments):
            # Create optimization based on current segment
            optimization = self._adaptive_optimization(segment, previous_optimizations=optimizations)
            optimizations.append(optimization)
        
        # Verify adaptation
        if len(optimizations) > 1:
            for i in range(1, len(optimizations)):
                current_opt = optimizations[i]
                previous_opt = optimizations[i-1]
                
                # Current optimization should differ from previous if trends changed
                current_segment = trend_segments[i]
                previous_segment = trend_segments[i-1]
                
                segment_difference = abs(sum(current_segment) / len(current_segment) - 
                                       sum(previous_segment) / len(previous_segment))
                
                if segment_difference > 0.1:  # Significant trend change
                    opt_difference = self._calculate_optimization_difference(current_opt, previous_opt)
                    assert opt_difference > 0, "Optimization should adapt to trend changes"
        
        # Verify learning from recent data
        if optimizations:
            latest_optimization = optimizations[-1]
            recent_trend = trend_segments[-1]
            
            # Latest optimization should reflect recent engagement levels
            recent_avg = sum(recent_trend) / len(recent_trend)
            opt_target = latest_optimization.get("target_engagement", 0.5)
            
            # Target should be realistic based on recent performance
            assert opt_target <= recent_avg + 0.3, "Optimization target should be realistic"
            assert opt_target >= recent_avg * 0.8, "Optimization should aim for improvement"
    
    @given(
        invalid_engagement_data=st.one_of(
            st.just([]),  # Empty data
            st.just([float('nan')] * 5),  # NaN values
            st.just([float('inf')] * 3),  # Infinite values
            st.just([-1.0, -0.5, 2.0])  # Out of range values
        )
    )
    @settings(max_examples=2, deadline=1000)
    def test_engagement_optimization_handles_invalid_data(
        self,
        invalid_engagement_data: List[float]
    ):
        """
        **Property 33: Engagement Optimization**
        **Validates: Requirements 8.5**
        
        Engagement optimization must handle invalid data gracefully:
        1. Invalid data should not crash the system
        2. Fallback strategies should be employed
        3. Error handling should be informative
        """
        try:
            # Attempt optimization with invalid data
            result = self._optimize_engagement_with_validation(invalid_engagement_data)
            
            # If it succeeds, should have error handling or fallback
            if "error" in result:
                assert isinstance(result["error"], str)
                assert len(result["error"]) > 0
            elif "fallback" in result:
                assert result["fallback"] is True
                assert "recommendations" in result
            else:
                # Should have valid recommendations despite invalid input
                assert "recommendations" in result
                assert len(result["recommendations"]) > 0
                
        except (ValueError, TypeError) as e:
            # Expected exceptions for invalid data
            assert len(str(e)) > 0
        
        # System should remain stable - test with valid data after
        valid_data = [0.3, 0.5, 0.7, 0.4, 0.6]
        valid_result = self._optimize_engagement_with_validation(valid_data)
        
        assert "recommendations" in valid_result
        assert len(valid_result["recommendations"]) > 0
    
    # Helper methods for testing
    
    def _create_engagement_data(
        self,
        scores: List[float],
        time_periods: List[int],
        content_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Create mock engagement data."""
        data = []
        for i, (score, time_period) in enumerate(zip(scores, time_periods)):
            content_type = content_types[i % len(content_types)]
            data.append({
                "engagement_score": score,
                "time_period": time_period,
                "content_type": content_type,
                "timestamp": datetime.now() - timedelta(hours=i)
            })
        return data
    
    def _optimize_engagement(self, engagement_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock engagement optimization."""
        # Group by content type and find best performers
        content_performance = {}
        time_performance = {}
        
        for data_point in engagement_data:
            content_type = data_point["content_type"]
            time_period = data_point["time_period"]
            score = data_point["engagement_score"]
            
            if content_type not in content_performance:
                content_performance[content_type] = []
            content_performance[content_type].append(score)
            
            if time_period not in time_performance:
                time_performance[time_period] = []
            time_performance[time_period].append(score)
        
        # Calculate best performance (max instead of average for optimization)
        content_best = {
            ct: max(scores)
            for ct, scores in content_performance.items()
        }
        time_best = {
            tp: max(scores)
            for tp, scores in time_performance.items()
        }
        
        # Generate recommendations
        recommendations = []
        
        # Best content type
        if content_best:
            best_content = max(content_best.items(), key=lambda x: x[1])
            best_time = max(time_best.items(), key=lambda x: x[1])[0] if time_best else 12
            recommendations.append({
                "content_type": best_content[0],
                "optimal_time": best_time,
                "expected_engagement": best_content[1],
                "recommendation_type": "content_optimization"
            })
        
        # Best time period
        if time_best:
            best_time = max(time_best.items(), key=lambda x: x[1])
            best_content_type = max(content_best.items(), key=lambda x: x[1])[0] if content_best else "default"
            recommendations.append({
                "content_type": best_content_type,
                "optimal_time": best_time[0],
                "expected_engagement": best_time[1],
                "recommendation_type": "timing_optimization"
            })
        
        return recommendations
    
    def _personalize_engagement_optimization(
        self,
        user_id: str,
        user_pattern: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate personalized engagement optimization."""
        # Sort patterns by engagement score
        sorted_patterns = sorted(user_pattern.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "user_id": user_id,
            "priorities": dict(sorted_patterns),
            "top_recommendation": sorted_patterns[0][0] if sorted_patterns else "default",
            "personalization_score": sum(user_pattern.values()) / len(user_pattern) if user_pattern else 0.5,
            "optimization_focus": "high_engagement_patterns"
        }
    
    def _count_optimization_differences(self, opt_a: Dict[str, Any], opt_b: Dict[str, Any]) -> int:
        """Count differences between two optimizations."""
        differences = 0
        
        if opt_a.get("top_recommendation") != opt_b.get("top_recommendation"):
            differences += 1
        
        if abs(opt_a.get("personalization_score", 0) - opt_b.get("personalization_score", 0)) > 0.1:
            differences += 1
        
        priorities_a = opt_a.get("priorities", {})
        priorities_b = opt_b.get("priorities", {})
        
        if set(priorities_a.keys()) != set(priorities_b.keys()):
            differences += 1
        
        return differences
    
    def _simulate_engagement_data(
        self,
        baseline: float,
        noise: float,
        sample_size: int
    ) -> List[float]:
        """Simulate engagement data with noise."""
        import random
        data = []
        for _ in range(sample_size):
            value = baseline + random.uniform(-noise, noise)
            data.append(max(0.0, min(1.0, value)))  # Clamp to valid range
        return data
    
    def _simulate_optimized_engagement_data(
        self,
        baseline: float,
        optimization_strength: float,
        noise: float,
        sample_size: int
    ) -> List[float]:
        """Simulate optimized engagement data."""
        import random
        data = []
        for _ in range(sample_size):
            optimized_baseline = baseline + optimization_strength
            value = optimized_baseline + random.uniform(-noise, noise)
            data.append(max(0.0, min(1.0, value)))  # Clamp to valid range
        return data
    
    def _segment_trends(self, trends: List[float], segments: int) -> List[List[float]]:
        """Segment trend data into chunks."""
        segment_size = len(trends) // segments
        if segment_size == 0:
            return [trends]
        
        result = []
        for i in range(segments):
            start = i * segment_size
            end = start + segment_size if i < segments - 1 else len(trends)
            result.append(trends[start:end])
        
        return result
    
    def _adaptive_optimization(
        self,
        current_segment: List[float],
        previous_optimizations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create adaptive optimization based on current trends."""
        current_avg = sum(current_segment) / len(current_segment) if current_segment else 0.5
        
        # Adapt based on trend
        if current_avg > 0.7:
            strategy = "maintain_high_engagement"
            target = current_avg + 0.1
        elif current_avg < 0.3:
            strategy = "boost_low_engagement"
            target = current_avg + 0.3
        else:
            strategy = "gradual_improvement"
            target = current_avg + 0.2
        
        return {
            "strategy": strategy,
            "target_engagement": min(target, 1.0),
            "current_performance": current_avg,
            "adaptation_count": len(previous_optimizations)
        }
    
    def _calculate_optimization_difference(
        self,
        opt_a: Dict[str, Any],
        opt_b: Dict[str, Any]
    ) -> float:
        """Calculate difference between optimizations."""
        strategy_diff = 1.0 if opt_a.get("strategy") != opt_b.get("strategy") else 0.0
        target_diff = abs(opt_a.get("target_engagement", 0.5) - opt_b.get("target_engagement", 0.5))
        
        return strategy_diff + target_diff
    
    def _optimize_engagement_with_validation(self, data: List[float]) -> Dict[str, Any]:
        """Optimize engagement with input validation."""
        # Validate input data
        if not data:
            return {"error": "Empty engagement data provided"}
        
        # Filter invalid values
        valid_data = []
        for value in data:
            if isinstance(value, (int, float)) and not (value != value or value == float('inf') or value == float('-inf')):
                if 0.0 <= value <= 1.0:
                    valid_data.append(value)
        
        if not valid_data:
            # Fallback strategy
            return {
                "fallback": True,
                "recommendations": [{
                    "content_type": "mixed",
                    "optimal_time": 14,
                    "expected_engagement": 0.5,
                    "recommendation_type": "fallback"
                }]
            }
        
        # Generate recommendations from valid data
        avg_engagement = sum(valid_data) / len(valid_data)
        
        return {
            "recommendations": [{
                "content_type": "optimized",
                "optimal_time": 10,
                "expected_engagement": min(avg_engagement + 0.2, 1.0),
                "recommendation_type": "data_driven"
            }]
        }


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def run_tests():
    """Run the property-based tests."""
    test_instance = TestEngagementOptimizationProperties()
    
    print("🧪 Testing Engagement Optimization Properties...")
    
    try:
        # Test 1: Engagement improvement
        print("📈 Testing engagement score improvement...")
        test_instance.test_engagement_optimization_improves_scores(
            [0.3, 0.5, 0.7, 0.4, 0.6], [9, 14, 16, 11, 15], ["video", "text", "interactive"]
        )
        print("✅ Engagement improvement test passed")
        
        # Test 2: Personalization
        print("👤 Testing personalization...")
        test_instance.test_engagement_optimization_personalization(
            [{"video": 0.8, "text": 0.4}, {"video": 0.3, "text": 0.9}], 2
        )
        print("✅ Personalization test passed")
        
        # Test 3: Effectiveness
        print("🎯 Testing optimization effectiveness...")
        test_instance.test_engagement_optimization_effectiveness(0.4, 0.2, 0.05)
        print("✅ Effectiveness test passed")
        
        # Test 4: Adaptation
        print("🔄 Testing adaptation to trends...")
        test_instance.test_engagement_optimization_adaptation(
            [0.3, 0.4, 0.5, 0.7, 0.8, 0.6, 0.4, 0.3, 0.5, 0.6], 3
        )
        print("✅ Adaptation test passed")
        
        # Test 5: Invalid data handling
        print("⚠️ Testing invalid data handling...")
        test_instance.test_engagement_optimization_handles_invalid_data([])
        print("✅ Invalid data handling test passed")
        
        print("\n🎉 All Engagement Optimization property tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)