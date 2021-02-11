pkill python
export PYTHONPATH=$PYTHONPATH:scripts/amd
pip install -e .
sh scripts/amd/trace_test.sh