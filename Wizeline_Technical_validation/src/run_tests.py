from unittest import main
import os

test_pipeline = main(module="test_pipeline", exit=False)
test_metrics = main(module="test_metrics", exit=False)
print(test_metrics.result.wasSuccessful)

with open(os.path.join('../tests', 'test_results.json'), 'w') as f:
    f.write(f'{{"test_pipeline.py results: ": {test_pipeline.result.wasSuccessful}, "test_metrics.py results: ": {test_metrics.result.wasSuccessful}}}')