import re
import numpy as np

str = open('test_result.txt').read()
result = re.findall('\ntp: (.*)  fp: (.*)  fn: (.*)\n', str)
result = np.array(result, dtype=np.int)
total = result.sum(axis=0)

precision = total[0] / (total[0] + total[1])
recall = total[0] / (total[0] + total[2])
fmeasure = 2 * precision * recall / (precision + recall)

print('tp: {0}  fp: {1}  fn: {2}'.format(total[0], total[1], total[2]))
print('Precision: {0}  Recall: {1}  Fmeasure: {2}'.format(precision, recall, fmeasure))
