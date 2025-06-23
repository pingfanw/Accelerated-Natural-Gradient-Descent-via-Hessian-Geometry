import os
import csv

# Prepare the CSV file for logging
def prepare_csv(log_path,optimizer,args):
    path = log_path+'/'+optimizer.lower()
    if not os.path.exists(path):
        os.makedirs(path)
    csv_ = open(path+'/'+optimizer.lower()+'_loss_dim'+str(args.m)+str(args.r)+'.csv','a+',newline='')
    csv_writer = csv.writer(csv_)
    return csv_, csv_writer

# Write a single row to the CSV file
def write_csv(csv_, csv_writer, head=False,**kwargs):
    if head:
        csv_writer.writerow(['Iteration','Loss'])
        csv_.flush()
    else:
        csv_writer.writerow([kwargs['iter'], kwargs['loss']])
        csv_.flush()