"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import argparse
import json

import pandas as pd


def load_submission(submission_file):
    # Load the submission CSV file
    submission = pd.read_csv(submission_file)
    return submission


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_file', type=str)
    args = parser.parse_args()

    # Load submission file
    submission = load_submission(args.submission_file)

    # Compute mean BLEU score
    result = {'BLEU Score': submission['bleu_score'].mean()}
    print(json.dumps(result))

if __name__ == '__main__':
    main()
