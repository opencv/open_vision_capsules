# Capsule Benchmark
This tool is used to produce benchmark result graphs for one or more capsules. 

The tool will benchmark the capsule at different levels of parallelism. 
Parallelism is defined as how many concurrent requests are being sent to the
capsule at any given time. For example, a parallelism of 3 is equivalent to 
3 threads pushing requests to the capsule as fast as they can be completed. 

The reason that parallelism is a focus on this benchmark is because many deep 
learning algorithms benefit from 'batching' requests. That is, sending multiple
images or inputs to the GPU / other devices concurrently. 

## Usage

### Install Dependencies
First, install the `vcap` and `vcap_utils` libraries. Then run:
```
pip3 install -r requirements.txt
```

### Running the script

Store one or more unpackaged capsules in a directory, in this example called
`capsules`. Then run the script and point it to the directory with the
capsule(s).
```
cd tools/capsule_benchmark
python3 main.py --capsule-dir /path/to/capsules/
```

The parallelism (x axis) can be changed by varying the `--parallelism`.
By default, it runs tests with 1, 2, 3, 5, and 10 parallelism. This can be
changed, for example, `--parallelism 1 10 100` will run 3 tests with 1, 10, 
and 100 as the parallelism parameters.
 
The number of samples can also be adjusted. `--num-samples 100` will run 100
samples per test. 

After running, an `output.html` graph will be in the current directory, 
also configurable via `--output-graph`.
