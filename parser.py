'''
Parser

Read lists of features from the UCI database (http://archive.ics.uci.edu/ml/datasets/Adult),
written in the format of `feature_list.txt`. Shows which numbers correspond to which feature values.
'''

# Queue of lines to write
to_write = []

# Read feature values
with open('feature_list.txt', mode='r') as f:
    for line in f.readlines():
        line_arr = line.split(':')
        feature_name = line_arr[0]
        feature_values_str = line_arr[1]
        feature_values = []
        if feature_values_str.strip().startswith('continuous'):
            feature_values = [
                '{0} first quartile'.format(feature_name),
                '{0} second quartile'.format(feature_name),
                '{0} third quartile'.format(feature_name),
                '{0} fourth quartile'.format(feature_name),
                '{0} not given'.format(feature_name)
            ]
        else:
            feature_values = feature_values_str.split(',')
            for i, value in enumerate(feature_values):
                feature_values[i] = '{0} {1}'.format(feature_name, value.strip().lower().replace('.',''))
        for value in feature_values:
            to_write.append(value)
    f.close()

# Write list of feature values
with open('feature_key.txt', mode='w') as f:
    for i, line in enumerate(to_write):
        f.write(str(i + 1) + ': '  + line + '\n')
    f.close()