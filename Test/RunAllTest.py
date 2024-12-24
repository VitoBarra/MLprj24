import unittest

if __name__ == "__main__":
    # Discover and load all test cases from the current directory and subdirectories
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='Test/', pattern='test_*.py')

    # Run the collected tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)