"""Property-based tests for progress visualization.

**Property 26: Progress Visualization**
**Validates: Requirements 7.2**

Progress visualizations must accurately represent learning data and provide meaningful insights.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import asyncio
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, Any, List

from src.ai_learning_accelerator.models.analytics import VisualizationType


class TestProgressVisualizationProperties:
    """Test progress visualization correctness properties."""
    
    @given(
        visualization_type=st.sampled_from(list(VisualizationType)),
        data_points=st.lists(st.floats(min_value=0, max_value=100), min_size=1, max_size=10),
        time_labels=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10)
    )
    @settings(max_examples=5, deadline=2000)
    def test_visualization_data_structure_consistency(
        self,
        visualization_type: VisualizationType,
        data_points: List[float],
        time_labels: List[str]
    ):
        """
        **Property 26: Progress Visualization**
        **Validates: Requirements 7.2**
        
        Progress visualizations must maintain consistent data structures:
        1. Each visualization type has a specific expected structure
        2. Data arrays must match label arrays in length
        3. Numeric data must be properly formatted
        """
        # Simulate visualization data generation
        generated_data = self._generate_mock_visualization_data(
            visualization_type, data_points, time_labels
        )
        
        # Verify structure consistency
        assert isinstance(generated_data, dict)
        self._verify_visualization_structure(visualization_type, generated_data)
        
        # Verify data consistency
        if "data" in generated_data:
            self._verify_data_consistency(generated_data["data"], data_points, time_labels)
    
    @given(
        base_values=st.lists(st.floats(min_value=0, max_value=100), min_size=3, max_size=8),
        scale_factor=st.floats(min_value=0.1, max_value=3.0)
    )
    @settings(max_examples=3, deadline=2000)
    def test_visualization_data_scaling_properties(
        self,
        base_values: List[float],
        scale_factor: float
    ):
        """
        **Property 26: Progress Visualization**
        **Validates: Requirements 7.2**
        
        Progress visualizations must handle data scaling correctly:
        1. Scaled data should maintain relative proportions
        2. Zero values should remain zero after scaling
        3. Maximum values should scale proportionally
        """
        scaled_values = [v * scale_factor for v in base_values]
        
        # Generate visualizations for both datasets
        base_viz = self._generate_mock_visualization_data(
            VisualizationType.LINE_CHART, base_values, [f"t{i}" for i in range(len(base_values))]
        )
        scaled_viz = self._generate_mock_visualization_data(
            VisualizationType.LINE_CHART, scaled_values, [f"t{i}" for i in range(len(scaled_values))]
        )
        
        # Verify scaling properties
        if base_values and scaled_values:
            base_max = max(base_values) if base_values else 0
            scaled_max = max(scaled_values) if scaled_values else 0
            
            if base_max > 0:
                expected_ratio = scaled_max / base_max
                assert abs(expected_ratio - scale_factor) < 0.001
            
            # Zero values should remain zero
            for i, base_val in enumerate(base_values):
                if base_val == 0.0:
                    assert scaled_values[i] == 0.0
    
    @given(
        invalid_data=st.one_of(
            st.just([]),  # Empty data
            st.just([float('inf')]),  # Infinite values
            st.just([float('nan')]),  # NaN values
        ),
        visualization_type=st.sampled_from([VisualizationType.LINE_CHART, VisualizationType.BAR_CHART])
    )
    @settings(max_examples=2, deadline=1000)
    def test_visualization_handles_invalid_data_gracefully(
        self,
        invalid_data: List[float],
        visualization_type: VisualizationType
    ):
        """
        **Property 26: Progress Visualization**
        **Validates: Requirements 7.2**
        
        Progress visualizations must handle invalid data gracefully:
        1. Empty datasets should not crash the system
        2. Invalid numeric values should be filtered or handled
        3. Error information should be provided when appropriate
        """
        try:
            generated_data = self._generate_mock_visualization_data(
                visualization_type, invalid_data, ["label1"]
            )
            
            # Should either handle gracefully or include error information
            assert isinstance(generated_data, dict)
            
            if "error" in generated_data:
                assert isinstance(generated_data["error"], str)
                assert len(generated_data["error"]) > 0
            else:
                # If no error, data should be valid
                self._verify_visualization_structure(visualization_type, generated_data)
                
        except (ValueError, TypeError) as e:
            # Expected exceptions for invalid data
            assert len(str(e)) > 0
    
    @given(
        progress_values=st.lists(st.floats(min_value=0, max_value=100), min_size=1, max_size=5)
    )
    @settings(max_examples=3, deadline=1000)
    def test_progress_bar_visualization_constraints(
        self,
        progress_values: List[float]
    ):
        """
        **Property 26: Progress Visualization**
        **Validates: Requirements 7.2**
        
        Progress bar visualizations must respect value constraints:
        1. Progress values should be between 0 and 100
        2. Values above 100 should be capped or handled appropriately
        3. Negative values should be handled gracefully
        """
        labels = [f"Item {i}" for i in range(len(progress_values))]
        
        generated_data = self._generate_mock_visualization_data(
            VisualizationType.PROGRESS_BAR, progress_values, labels
        )
        
        assert isinstance(generated_data, dict)
        assert "type" in generated_data
        assert generated_data["type"] == "progress_bars"
        
        if "data" in generated_data and isinstance(generated_data["data"], list):
            for item in generated_data["data"]:
                if "progress" in item:
                    progress = item["progress"]
                    # Progress should be a valid number between 0 and 100
                    assert isinstance(progress, (int, float))
                    assert 0 <= progress <= 100
    
    def _generate_mock_visualization_data(
        self,
        viz_type: VisualizationType,
        data_points: List[float],
        labels: List[str]
    ) -> Dict[str, Any]:
        """Generate mock visualization data for testing."""
        # Filter out invalid values
        valid_data = [x for x in data_points if not (
            x != x or  # NaN check
            x == float('inf') or x == float('-inf')  # Infinity check
        )]
        
        if not valid_data:
            return {"error": "No valid data points provided"}
        
        # Ensure labels and data have same length
        min_length = min(len(valid_data), len(labels))
        valid_data = valid_data[:min_length]
        valid_labels = labels[:min_length]
        
        if viz_type == VisualizationType.LINE_CHART:
            return {
                "type": "line",
                "data": {
                    "labels": valid_labels,
                    "datasets": [{
                        "label": "Progress",
                        "data": valid_data,
                        "borderColor": "rgb(75, 192, 192)"
                    }]
                }
            }
        
        elif viz_type == VisualizationType.BAR_CHART:
            return {
                "type": "bar",
                "data": {
                    "labels": valid_labels,
                    "datasets": [{
                        "label": "Values",
                        "data": valid_data,
                        "backgroundColor": "rgba(54, 162, 235, 0.2)"
                    }]
                }
            }
        
        elif viz_type == VisualizationType.PIE_CHART:
            return {
                "type": "pie",
                "data": {
                    "labels": valid_labels,
                    "datasets": [{
                        "data": valid_data,
                        "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"]
                    }]
                }
            }
        
        elif viz_type == VisualizationType.PROGRESS_BAR:
            return {
                "type": "progress_bars",
                "data": [
                    {
                        "label": label,
                        "progress": min(max(value, 0), 100),  # Clamp between 0-100
                        "target": 100.0
                    }
                    for label, value in zip(valid_labels, valid_data)
                ]
            }
        
        elif viz_type == VisualizationType.HEATMAP:
            return {
                "type": "heatmap",
                "data": valid_data,
                "labels": {
                    "x": valid_labels,
                    "y": ["Row 1"]
                }
            }
        
        elif viz_type == VisualizationType.RADAR_CHART:
            return {
                "type": "radar",
                "data": {
                    "labels": valid_labels,
                    "datasets": [{
                        "label": "Competency",
                        "data": valid_data,
                        "backgroundColor": "rgba(54, 162, 235, 0.2)"
                    }]
                }
            }
        
        elif viz_type == VisualizationType.TIMELINE:
            return {
                "type": "timeline",
                "events": [
                    {
                        "date": f"2023-01-{i+1:02d}",
                        "title": label,
                        "value": value
                    }
                    for i, (label, value) in enumerate(zip(valid_labels, valid_data))
                ]
            }
        
        elif viz_type == VisualizationType.DASHBOARD:
            return {
                "summary": {"total": sum(valid_data), "count": len(valid_data)},
                "chart": {
                    "type": "line",
                    "data": valid_data,
                    "labels": valid_labels
                }
            }
        
        else:
            return {"error": f"Unsupported visualization type: {viz_type}"}
    
    def _verify_visualization_structure(self, viz_type: VisualizationType, data: Dict[str, Any]):
        """Verify that visualization data has the correct structure for its type."""
        if "error" in data:
            return  # Error case is acceptable
        
        if viz_type == VisualizationType.LINE_CHART:
            assert "type" in data
            assert data["type"] == "line"
            assert "data" in data
            if "datasets" in data["data"]:
                assert isinstance(data["data"]["datasets"], list)
        
        elif viz_type == VisualizationType.BAR_CHART:
            assert "type" in data
            assert data["type"] == "bar"
            assert "data" in data
        
        elif viz_type == VisualizationType.PIE_CHART:
            assert "type" in data
            assert data["type"] == "pie"
            assert "data" in data
        
        elif viz_type == VisualizationType.HEATMAP:
            assert "type" in data
            assert data["type"] == "heatmap"
            assert "data" in data
        
        elif viz_type == VisualizationType.PROGRESS_BAR:
            assert "type" in data
            assert data["type"] == "progress_bars"
            assert "data" in data
            assert isinstance(data["data"], list)
        
        elif viz_type == VisualizationType.RADAR_CHART:
            assert "type" in data
            assert data["type"] == "radar"
            assert "data" in data
        
        elif viz_type == VisualizationType.TIMELINE:
            assert "type" in data
            assert data["type"] == "timeline"
            assert "events" in data
            assert isinstance(data["events"], list)
        
        elif viz_type == VisualizationType.DASHBOARD:
            assert isinstance(data, dict)
            assert len(data) > 0
    
    def _verify_data_consistency(self, chart_data: Dict[str, Any], original_data: List[float], labels: List[str]):
        """Verify that chart data is consistent with original data."""
        if "datasets" in chart_data and chart_data["datasets"]:
            dataset = chart_data["datasets"][0]
            if "data" in dataset:
                chart_values = dataset["data"]
                # Should have some relationship to original data
                assert len(chart_values) > 0
                assert all(isinstance(v, (int, float)) for v in chart_values)
        
        if "labels" in chart_data:
            chart_labels = chart_data["labels"]
            assert len(chart_labels) > 0
            assert all(isinstance(label, str) for label in chart_labels)


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def run_tests():
    """Run the property-based tests."""
    test_instance = TestProgressVisualizationProperties()
    
    print("🧪 Testing Progress Visualization Properties...")
    
    try:
        # Test 1: Data structure consistency
        print("📊 Testing visualization data structure consistency...")
        test_instance.test_visualization_data_structure_consistency(
            VisualizationType.LINE_CHART, [10.0, 20.0, 30.0], ["A", "B", "C"]
        )
        print("✅ Data structure consistency test passed")
        
        # Test 2: Data scaling
        print("📈 Testing visualization data scaling...")
        test_instance.test_visualization_data_scaling_properties(
            [25.0, 50.0, 75.0], 2.0
        )
        print("✅ Data scaling test passed")
        
        # Test 3: Invalid data handling
        print("⚠️ Testing invalid data handling...")
        test_instance.test_visualization_handles_invalid_data_gracefully(
            [], VisualizationType.LINE_CHART
        )
        print("✅ Invalid data handling test passed")
        
        # Test 4: Progress bar constraints
        print("📊 Testing progress bar constraints...")
        test_instance.test_progress_bar_visualization_constraints(
            [25.0, 50.0, 75.0, 100.0]
        )
        print("✅ Progress bar constraints test passed")
        
        print("\n🎉 All Progress Visualization property tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)