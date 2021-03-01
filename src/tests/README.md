# Tests
The tests in this folder are our tool to check if everything is working as intended and to pick any errors that might
appear when introducing new features. They can also be used to make sure all dependencies are available before running
any long experiments.

## Running tests

### From console
Type the following code in your console:
```bash
cd src/
py.test -s tests/
```

### Running tests in parallel
As the amount of tests grow, it can be faster to run them in parallel since they can take few minutes each.
Pytest can support it by using the `pytest-xdist` plugin, e.g. running all tests with 5 workers:
```bash 
cd src/
py.test -n 5 tests/
```
And for a more verbose output:
```bash 
cd src/
py.test -sv -n 5 tests/
```
**_Warning:_** It is recommended to run a single test without parallelization the first time, since the first thing that
our test will do is download the dataset (MNIST). If ran in parallel they can start downloading it in multiple workers
at the same time.

### From your IDE (PyCharm, VSCode, ...)
`py.tests` are well supported. It's usually enough to select `py.test` as a framework, right click on a test file or
directory and select the option to run pytest tests (not as a python script!).
