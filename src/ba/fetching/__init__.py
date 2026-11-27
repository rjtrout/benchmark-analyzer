"""Fetching: everything ``ba fetch`` reads from outside, and how it is stored.

All of it is public, and read with no account:

    fetch.py        what `ba fetch` does, step by step
    benchmarks.py   a benchmark's test cases, from its dataset on the Hugging Face Hub
    loaders.py      a dataset's rows, from the Parquet files the Hub stores
    leaderboard.py  a model's transcript files, from the leaderboard's public bucket
    manifests.py    a transcript file, and joining it to a benchmark's test case list

Nothing outside this folder downloads anything. The rest of the package works on what is
stored, and only the ``ba fetch`` command imports from here.
"""
