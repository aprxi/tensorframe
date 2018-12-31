'''
Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).

TensorFrame support test functions: profiles
'''

import os
import re
import yaml
import json
import stat
import time


def yaml_load(filename):
    with open(filename, 'r') as f:
        return yaml.load(f.read())


def sort_nicely(input_list, reverse=False):
    '''
    Sort the given list in the way that humans expect.
    Source: https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    '''
    assert(isinstance(input_list, list))
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(input_list, key=alphanum_key, reverse=reverse)


def filter_listdir_age(directory, age_seconds=86400):
    '''
    Retrieve most recent (sub)directory from $(directory), no older than $(age_seconds)"
    '''
    assert isinstance(directory, str) 
    assert isinstance(age_seconds, int)

    if not os.path.isdir(directory):
        return []

    current_time = time.time().__int__() 

    matched = {fd: os.stat(directory + '/' + fd) for fd in os.listdir(directory) if fd[0] != '.'}
    filtered = [
        fd for fd, fd_stat in matched.items()
        if stat.S_ISDIR(fd_stat.st_mode)
        and int(current_time - fd_stat.st_ctime.__int__()) < age_seconds
    ]
    return sort_nicely(filtered, reverse=True)


def filter_listfiles_ext(directory, ext='yml'):
    '''
    Retrieve files within directory. Ending on $(ext), skipping hidden.
    '''
    assert isinstance(directory, str) 
    assert isinstance(ext, str)
    
    if not os.path.isdir(directory):
        return []

    matched = {
        fd: os.stat(directory + '/' + fd) for fd in os.listdir(directory)
        if re.match('^[-\w]*\.' + ext + '$', fd)
    }
    filtered = [
        fd for fd, fd_stat in matched.items()
        if stat.S_ISREG(fd_stat.st_mode)
    ]
    return sort_nicely(filtered)


def load_testruns(filename):
    configuration = yaml_load(filename)

    assert(isinstance(configuration, dict))
    assert(isinstance(configuration.get('tests'), list))
    assert(isinstance(configuration.get('defaults'), dict))

    includes = configuration['defaults'].get('include')
    if isinstance(includes, list):
        include_files = [
            os.path.dirname(os.path.realpath(__file__)) + '/' + fn
            for fn in includes if isinstance(fn, str)]
        for fn in include_files:
            configuration['defaults'].update(yaml_load(fn))

    assert(isinstance(configuration['defaults'].get('datatypes'), dict))

    if not isinstance(configuration['defaults'].get('rows_no'), int):
        configuration['defaults']['rows_no'] = 10
    if not isinstance(configuration['defaults'].get('columns_no'), int):
        configuration['defaults']['columns_no'] = 4

    group_name = re.sub('\.yml$', '', os.path.basename(filename))
    assert(isinstance(group_name, str))

    def _update_testrun(testrun):
        tr_name = group_name + '__' + testrun.get('name')
        tr_columns = testrun.get('columns')

        assert(isinstance(tr_name, str))

        default_columns_no = testrun.get('columns_no')
        if not isinstance(default_columns_no, int):
            default_columns_no = configuration['defaults']['columns_no']

        default_rows_no = testrun.get('rows_no')
        if not isinstance(default_rows_no, int):
            default_rows_no = configuration['defaults']['rows_no']

        if isinstance(tr_columns, str):
            # Assume string to be datatype
            column_list = [{'datatype': tr_columns}]

        elif isinstance(tr_columns, list):
            column_list = tr_columns
        else:
            raise ValueError('_update_testrun() columns configuration incorrect')
       
        column_list_ret = []
        for column_group in column_list:
            datatype = column_group.get('datatype')
            columns_no = column_group.get('columns_no')

            assert(isinstance(datatype, str))
            datatype_config = configuration['defaults']['datatypes'].get(datatype)

            # Verify datatype configuration
            assert(isinstance(datatype_config, dict))
            for required_int_value in ['min_value', 'max_value']:
                assert(isinstance(datatype_config.get(required_int_value), int))

            if not isinstance(columns_no, int):
                columns_no = default_columns_no 

            column_list_ret = column_list_ret + [{
                    'name': 'col_' + str(column_id),
                    'datatype': datatype,
                    'min_value': datatype_config['min_value'],
                    'max_value': datatype_config['max_value'],
                } for column_id in range(len(column_list_ret), len(column_list_ret) + columns_no)
            ]
                           

        return {'name': tr_name, 'columns': column_list_ret, 'rows_no': default_rows_no}

    runs = [
        _update_testrun(testrun)
        for testrun in configuration.get('tests')
        if isinstance(testrun, dict)
    ]

    return runs
