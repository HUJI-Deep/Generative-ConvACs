#!/usr/bin/env python
import os, agate, sys

def percentages(column, precision=2):
    return (column, agate.Formula(agate.Text(), lambda r: ('%%.%df' % precision) % (r[column]*100)))

def summary(table):
    table = table.where(lambda x: x['optimization_done'] == True)
    accuracies = table.select(['hyper__rect_size', 'hyper__num_rects','test_results0_0_accuracy'])
    accuracies = accuracies.rename({'test_results0_0_accuracy': 'accuracy', 'hyper__rect_size': 'Width', 'hyper__num_rects': 'Number of Rectangles'})
    accuracies = accuracies.order_by(['Width', 'Number of Rectangles'])
    accuracies = accuracies.compute([percentages('accuracy')], replace=True)
    accuracies.print_table()

if __name__ == '__main__':
    csv_files = filter(lambda x: x.endswith('.csv') and 'unique' not in x, os.listdir('.'))
    if len(csv_files) == 0:
        print 'Cannot find results file. Make sure to execute run.py first.'
        sys.exit(-1)
    csv_file = csv_files[0]
    table = agate.Table.from_csv(csv_file)
    summary(table)
