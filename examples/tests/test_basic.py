"""
Test basic example functionality
"""
import pytest


def test_imports():
    """Test that basic example can be imported"""
    try:
        from q_store_examples import basic_example
        assert basic_example is not None
    except ImportError:
        pytest.skip("q_store_examples not installed")


def test_package_version():
    """Test package version is set"""
    try:
        import q_store_examples
        assert hasattr(q_store_examples, '__version__')
        assert q_store_examples.__version__ == "0.1.0"
    except ImportError:
        pytest.skip("q_store_examples not installed")
