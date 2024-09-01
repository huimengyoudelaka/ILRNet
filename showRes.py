import scipy.io as scio

import csv

def saveAsCsv(savePath,res):
    csv_file = open(savePath, 'w', newline='', encoding='gbk')
    writer = csv.writer(csv_file)
    writer.writerow(['psnr', 'ssim', 'sam'])
    for row in res:
        writer.writerow(row)
    csv_file.close()

# dataFile = '15dB_Res.mat'
# data = scio.loadmat(dataFile)
# res = data['res_arr']
# saveAsCsv('15dB_Res.csv',res)