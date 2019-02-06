"""
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).

Test core functionality of TensorFrame
"""

import os
import re

import unittest

import numpy as np
import pyarrow as pa

import tensorframe
import test_profiles


class TestLoadCSV(unittest.TestCase):
    """
    Test run of generating, loading and summing a CSV file.
    """

    def setUp(self):
        """
        Generate the test csv files to load with data based on test profiles
        """
        self.tmpdir = os.environ['HOME'] + '/.tensorframe/test'
        self.testfiles = []

        # Override default if set (through environment)
        if isinstance(os.environ.get('TEST_PROFILE_DIR'), str):
            self.configdir = os.environ.get('TEST_PROFILE_DIR')
        else:
            #Default test case
            self.configdir = os.path.dirname(os.path.realpath(__file__)) + '/functions/loadcsv'

        assert os.path.isdir(self.configdir)

        if not os.path.isdir(self.tmpdir):
            os.makedirs(self.tmpdir)

        testrun_batches = {
            re.sub(r'\.yml$', '', fn): test_profiles.load_testruns(self.configdir + '/' + fn)
            for fn in test_profiles.filter_listfiles_ext(self.configdir, ext='yml')
        }

        # base memory need on requirement of largest test case
        max_bytes = sorted([
            column_group['total_bytes'] for sub_batch in
            [batch for batch in list(testrun_batches.values())]
            for column_group in sub_batch])[-1]

        # initialise a tensorframe to be used in test runs
        self.frame = tensorframe.TensorFrame(memory=max_bytes)

        for batch_name, batch_run in testrun_batches.items():
            outputdir = self.tmpdir + '/' + batch_name
            for csv_contents in batch_run:
                self.testfiles.append(self.generate(outputdir, csv_contents))

        assert self.testfiles.__len__() > 0


    def generate(self, outputdir, csv_contents):
        """Initialise tensorframe with (pseudo-)random data based on schema,
        calculate meta and write to file."""
        assert isinstance(csv_contents, dict)
        assert isinstance(csv_contents.get('name'), str)
        assert isinstance(outputdir, str)

        outputfile = outputdir + '/' + csv_contents['name'] + '.csv'
        datatypes = tensorframe.arrow.convert_dtype_from_stringlist(
            [column['datatype'] for column in csv_contents['columns']])

        if not (os.path.isfile(outputfile) and os.path.isfile(outputfile + '.meta')):
            print('\n[TASK] (RE-)GENERATING CSV: ' + outputfile)
            if not os.path.isdir(outputdir):
                os.makedirs(outputdir)
            #self.frame.random(csv_contents)
            self.frame.random(csv_contents)
            self.frame.write_csv(outputfile, write_meta=True)

        return {'filename': outputfile, 'datatypes': datatypes}


    def test_read_csv(self):
        """Load the generated csv and test if its correct."""

        for testfile in self.testfiles:
            outputfile = testfile['filename']
            dtype = testfile['datatypes']
            #pabatch = self.frame.read_csv(outputfile, dtype=dtype).arrow_recordbatch()
            pabatch = self.frame.read_csv(outputfile, dtype=dtype).to_arrow()

            print(type(pabatch))
            # check a sample
            print(pabatch[0])

            # print schema (todo: automatically verify with that of input dtype)
            print(pabatch.schema)
            
            # check if all columns add up properly
            print('column_sums:\n', self.frame.sum())


        # TODO: add additional verification. Most notably:
        # - sum verification (numpy sum vs. tf.array_sum)
        # - integrate the timing functions to check on performance regressions (/improvements)
