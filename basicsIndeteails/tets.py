import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
import sys

import warnings

warnings.filterwarnings("ignore")


# Define All required functions

def getIMSI(row):
    condition = ((row['time'] <= UeHandoverStartStats['Time'])
                 & (row['cellId'] == UeHandoverStartStats['SourceCellId'])
                 & (row['rnti'] == UeHandoverStartStats['RNTI']))
    result = UeHandoverStartStats[condition]
    if result.shape[0] != 0:
        return result.iloc[-1]['IMSI']
    condition = ((DlRlcStats['start'] <= row['time'])
                 & (DlRlcStats['end'] >= row['time'])
                 & (DlRlcStats['CellId'] == row['cellId'])
                 & (DlRlcStats['RNTI'] == row['rnti']))
    result = DlRlcStats[condition]
    if result.shape[0] != 0:
        return result.iloc[0]['IMSI']


def optimizer_sinr_mcs(row):
    final_cell = []
    for cell in range(2, 9):
        condition = ((RxPacketTrace['time'] >= row['Time'])
                     & (RxPacketTrace['time'] <= row['Time'] + 0.01)
                     & (RxPacketTrace['IMSI'] == row['IMSI'])
                     & (RxPacketTrace['cellId'] == cell))
        final_cell.append(RxPacketTrace[condition]['SINR(dB)'].values)
        final_cell.append(RxPacketTrace[condition]['mcs'].values)
    return final_cell[0], final_cell[2], final_cell[4], final_cell[6], final_cell[8], final_cell[10], final_cell[12], \
           final_cell[1], final_cell[3], final_cell[5], final_cell[7], final_cell[9], final_cell[11], final_cell[13],


def bin_sinr(x):
    return len(x[(x < -6)]), len(x[(x >= -6) & (x < 0)]), len(x[(x >= 0) & (x < 6)]), len(x[(x >= 6) & (x < 12)]), len(
        x[(x >= 12) & (x < 18)]), len(x[(x >= 18) & (x < 24)]), len(x[(x >= 24)])


def bin_mcs(x):
    return len(x[(x >= 0) & (x <= 4)]), len(x[(x >= 5) & (x <= 9)]), len(x[(x >= 10) & (x <= 14)]), len(
        x[(x >= 15) & (x <= 19)]), len(x[(x >= 20) & (x <= 24)]), len(x[(x >= 25) & (x <= 29)])


def optimize(row):
    val_lst = [0, 0, 0, 0, 0, 0, 0]
    if row['cellId'] == 2:
        val_lst[0] = row['mcs']
    elif row['cellId'] == 3:
        val_lst[1] = row['mcs']
    elif row['cellId'] == 4:
        val_lst[2] = row['mcs']
    elif row['cellId'] == 5:
        val_lst[3] = row['mcs']
    elif row['cellId'] == 6:
        val_lst[4] = row['mcs']
    elif row['cellId'] == 7:
        val_lst[5] = row['mcs']
    elif row['cellId'] == 8:
        val_lst[6] = row['mcs']
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def optimize_summation(row):
    window_size = 0.01
    val_lst = [0, 0, 0, 0, 0, 0, 0]
    if row['cellId'] == 2:
        val_lst[0] = ((row['tbSize'] * 8) / window_size) / 1000  # ((row['tbSize']*8)/window_size)/1000 #(kbps)
    elif row['cellId'] == 3:
        val_lst[1] = ((row['tbSize'] * 8) / window_size) / 1000
    elif row['cellId'] == 4:
        val_lst[2] = ((row['tbSize'] * 8) / window_size) / 1000
    elif row['cellId'] == 5:
        val_lst[3] = ((row['tbSize'] * 8) / window_size) / 1000
    elif row['cellId'] == 6:
        val_lst[4] = ((row['tbSize'] * 8) / window_size) / 1000
    elif row['cellId'] == 7:
        val_lst[5] = ((row['tbSize'] * 8) / window_size) / 1000
    elif row['cellId'] == 8:
        val_lst[6] = ((row['tbSize'] * 8) / window_size) / 1000
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def max_val(row, rx_df):
    return rx_df[(rx_df['Time'] == row['Time']) & (rx_df['IMSI'] == row['IMSI'])].max().values


def drbthpdl(row):
    condition = ((row['cellId'] == DlRlcStats['CellId'])
                 & (row['IMSI'] == DlRlcStats['IMSI'])
                 & (row['Time'] == DlRlcStats['start']))
    result = DlRlcStats[condition]
    if not result.empty:
        return (result['TxBytes'] * 8 / row['symbol#']).values[0]
    return 0


def drbthpdl_optimize(row):
    val_lst = [0, 0, 0, 0, 0, 0, 0]
    if row['cellId'] == 2:
        val_lst[0] = row['DRB.IpThpDl.UEID']
    elif row['cellId'] == 3:
        val_lst[1] = row['DRB.IpThpDl.UEID']
    elif row['cellId'] == 4:
        val_lst[2] = row['DRB.IpThpDl.UEID']
    elif row['cellId'] == 5:
        val_lst[3] = row['DRB.IpThpDl.UEID']
    elif row['cellId'] == 6:
        val_lst[4] = row['DRB.IpThpDl.UEID']
    elif row['cellId'] == 7:
        val_lst[5] = row['DRB.IpThpDl.UEID']
    elif row['cellId'] == 8:
        val_lst[6] = row['DRB.IpThpDl.UEID']
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def PdcpPduNbrDl_optimize(row):
    val_lst = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    if row['CellId'] == 2:
        val_lst[0] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    elif row['CellId'] == 3:
        val_lst[1] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    elif row['CellId'] == 4:
        val_lst[2] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    elif row['CellId'] == 5:
        val_lst[3] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    elif row['CellId'] == 6:
        val_lst[4] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    elif row['CellId'] == 7:
        val_lst[5] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    elif row['CellId'] == 8:
        val_lst[6] = row['DRB.PdcpPduNbrDl.Qos.UEID']
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def optimize_IPTimeDL(row):
    val_lst = [0, 0, 0, 0, 0, 0, 0]
    if row['cellId'] == 2:
        val_lst[0] = row['symbol#']
    elif row['cellId'] == 3:
        val_lst[1] = row['symbol#']
    elif row['cellId'] == 4:
        val_lst[2] = row['symbol#']
    elif row['cellId'] == 5:
        val_lst[3] = row['symbol#']
    elif row['cellId'] == 6:
        val_lst[4] = row['symbol#']
    elif row['cellId'] == 7:
        val_lst[5] = row['symbol#']
    elif row['cellId'] == 8:
        val_lst[6] = row['symbol#']
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def optimize_PdcpPduVolumeDL_Filter(row):
    val_lst = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    if row['CellId'] == 2:
        val_lst[0] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    elif row['CellId'] == 3:
        val_lst[1] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    elif row['CellId'] == 4:
        val_lst[2] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    elif row['CellId'] == 5:
        val_lst[3] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    elif row['CellId'] == 6:
        val_lst[4] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    elif row['CellId'] == 7:
        val_lst[5] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    elif row['CellId'] == 8:
        val_lst[6] = row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def get_serving_cell(row):
    return \
    RxPacketTrace[(row['IMSI'] == RxPacketTrace['IMSI']) & (row['time'] == RxPacketTrace['time'])]['cellId'].values[0]


def optimize_UE_sp_PRB_schedule(row):
    val_lst = [0, 0, 0, 0, 0, 0, 0]
    if row['cellId'] == 2:
        val_lst[0] = (row['symbol#'] / (14 * 40)) * 139
    elif row['cellId'] == 3:
        val_lst[1] = (row['symbol#'] / (14 * 40)) * 139
    elif row['cellId'] == 4:
        val_lst[2] = (row['symbol#'] / (14 * 40)) * 139
    elif row['cellId'] == 5:
        val_lst[3] = (row['symbol#'] / (14 * 40)) * 139
    elif row['cellId'] == 6:
        val_lst[4] = (row['symbol#'] / (14 * 40)) * 139
    elif row['cellId'] == 7:
        val_lst[5] = (row['symbol#'] / (14 * 40)) * 139
    elif row['cellId'] == 8:
        val_lst[6] = (row['symbol#'] / (14 * 40)) * 139
    return val_lst[0], val_lst[1], val_lst[2], val_lst[3], val_lst[4], val_lst[5], val_lst[6]


def get_HO_Cell_Qual_SINR(row):
    HO_CellQual_RS_SINR_UEID_Cell2, HO_CellQual_RS_SINR_UEID_Cell3 = None, None
    HO_CellQual_RS_SINR_UEID_Cell4, HO_CellQual_RS_SINR_UEID_Cell4 = None, None
    HO_CellQual_RS_SINR_UEID_Cell6, HO_CellQual_RS_SINR_UEID_Cell5 = None, None
    HO_CellQual_RS_SINR_UEID_Cell8 = None

    condition = ((MmWaveSinrTime['Time'] <= row['Time'])
                 & (MmWaveSinrTime['IMSI'] == row['IMSI']))

    max_time = MmWaveSinrTime[condition]['Time'].max()

    result = MmWaveSinrTime[(MmWaveSinrTime['Time'] == max_time)
                            & (MmWaveSinrTime['IMSI'] == row['IMSI'])]
    for _, row in result.iterrows():
        if row['CellId'] == 2:
            HO_CellQual_RS_SINR_UEID_Cell2 = row['SINR']
        elif row['CellId'] == 3:
            HO_CellQual_RS_SINR_UEID_Cell3 = row['SINR']
        elif row['CellId'] == 4:
            HO_CellQual_RS_SINR_UEID_Cell4 = row['SINR']
        elif row['CellId'] == 5:
            HO_CellQual_RS_SINR_UEID_Cell5 = row['SINR']
        elif row['CellId'] == 6:
            HO_CellQual_RS_SINR_UEID_Cell6 = row['SINR']
        elif row['CellId'] == 7:
            HO_CellQual_RS_SINR_UEID_Cell7 = row['SINR']
        elif row['CellId'] == 8:
            HO_CellQual_RS_SINR_UEID_Cell8 = row['SINR']

    return [HO_CellQual_RS_SINR_UEID_Cell2, HO_CellQual_RS_SINR_UEID_Cell3, HO_CellQual_RS_SINR_UEID_Cell4,
            HO_CellQual_RS_SINR_UEID_Cell5, HO_CellQual_RS_SINR_UEID_Cell6, HO_CellQual_RS_SINR_UEID_Cell7,
            HO_CellQual_RS_SINR_UEID_Cell8]


def optimizer_sinr_mcs_cell(row):
    final_cell = []
    for cell in range(2, 9):
        condition = ((RxPacketTrace['time'] >= row['Time'])
                     & (RxPacketTrace['time'] <= row['Time'] + 0.01)
                     & (RxPacketTrace['cellId'] == cell))
        final_cell.append(RxPacketTrace[condition]['SINR(dB)'].values)
        final_cell.append(RxPacketTrace[condition]['mcs'].values)
    return final_cell[0], final_cell[2], final_cell[4], final_cell[6], final_cell[8], final_cell[10], final_cell[12], \
           final_cell[1], final_cell[3], final_cell[5], final_cell[7], final_cell[9], final_cell[11], final_cell[13],


def max_val_cell(row, rx_df):
    return rx_df[(rx_df['Time'] == row['Time'])].max().values


def initial_UE():
    ue_df_array = []
    time_window = 0.01
    times = np.arange(0.01, 6.001, time_window)
    for imsi in range(1, 50):
        ue_df = pd.DataFrame({'Time': times})
        ue_df.loc[:, 'IMSI'] = imsi
        ue_df_array.append(ue_df)
    UE = pd.DataFrame(columns=ue_df.columns)
    for dframe in ue_df_array:
        UE = UE.append(dframe, ignore_index=True)
    UE['Time'] = UE['Time'].apply(lambda x: round(x, 4))
    UE['IMSI'] = UE['IMSI'].astype('int32')
    return UE


def initial_rx_sinr_mcs():
    rx_imsi = RxPacketTrace['IMSI'].unique()
    rx_time = RxPacketTrace['Time'].unique()
    tmp_arry = []
    times = rx_time
    for imsi in rx_imsi:
        temp_df = pd.DataFrame({'Time': times})
        temp_df.loc[:, 'IMSI'] = imsi
        tmp_arry.append(temp_df)
    rx_sinr_mcs_df = pd.DataFrame(columns=temp_df.columns)

    for dframe in tmp_arry:
        rx_sinr_mcs_df = rx_sinr_mcs_df.append(dframe, ignore_index=True)
    rx_sinr_mcs_df['Time'] = rx_sinr_mcs_df['Time'].apply(lambda x: round(x, 4))
    rx_sinr_mcs_df['IMSI'] = rx_sinr_mcs_df['IMSI'].astype('int32')
    return rx_sinr_mcs_df


# def get_SINR_MCS_UEID(UE,rx_sinr_mcs_df):
#
#    df = pd.DataFrame(columns = ['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#                             'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'])
#    rx_sinr_mcs_df = pd.concat([rx_sinr_mcs_df,df])
#
#    #populate arrays as it's easy for later use
#    x = [np.array([],dtype='float64')]*len(rx_sinr_mcs_df)
#    rx_sinr_mcs_df[['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#                    'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8']] = \
#                    x,x,x,x,x,x,x,x,x,x,x,x,x,x
#    rx_sinr_mcs_df.loc[:,['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#                      'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8' ]] = \
#                        list(rx_sinr_mcs_df.loc[:,['Time','IMSI']].apply(lambda x:optimizer_sinr_mcs(x),axis=1))
#    UE = pd.merge(UE,rx_sinr_mcs_df,on=['Time','IMSI'],how='left')
#
#    #replace nan with empty array
#    for i in UE.columns:
#        UE[i].loc[UE[i].isnull()] = [np.array([],dtype='float64')]*len(UE[i].loc[UE[i].isnull()])
#
#    df = pd.DataFrame(columns = ['L1M.RS-SINR.Bin34.UEID.Cell2','L1M.RS-SINR.Bin46.UEID.Cell2','L1M.RS-SINR.Bin58.UEID.Cell2','L1M.RS-SINR.Bin70.UEID.Cell2',
#    'L1M.RS-SINR.Bin82.UEID.Cell2','L1M.RS-SINR.Bin94.UEID.Cell2','L1M.RS-SINR.Bin127.UEID.Cell2',
#    'L1M.RS-SINR.Bin34.UEID.Cell3','L1M.RS-SINR.Bin46.UEID.Cell3','L1M.RS-SINR.Bin58.UEID.Cell3','L1M.RS-SINR.Bin70.UEID.Cell3',
#    'L1M.RS-SINR.Bin82.UEID.Cell3','L1M.RS-SINR.Bin94.UEID.Cell3','L1M.RS-SINR.Bin127.UEID.Cell3',
#    'L1M.RS-SINR.Bin34.UEID.Cell4','L1M.RS-SINR.Bin46.UEID.Cell4','L1M.RS-SINR.Bin58.UEID.Cell4','L1M.RS-SINR.Bin70.UEID.Cell4',
#    'L1M.RS-SINR.Bin82.UEID.Cell4','L1M.RS-SINR.Bin94.UEID.Cell4','L1M.RS-SINR.Bin127.UEID.Cell4',
#    'L1M.RS-SINR.Bin34.UEID.Cell5','L1M.RS-SINR.Bin46.UEID.Cell5','L1M.RS-SINR.Bin58.UEID.Cell5','L1M.RS-SINR.Bin70.UEID.Cell5',
#    'L1M.RS-SINR.Bin82.UEID.Cell5','L1M.RS-SINR.Bin94.UEID.Cell5','L1M.RS-SINR.Bin127.UEID.Cell5',
#    'L1M.RS-SINR.Bin34.UEID.Cell6','L1M.RS-SINR.Bin46.UEID.Cell6','L1M.RS-SINR.Bin58.UEID.Cell6','L1M.RS-SINR.Bin70.UEID.Cell6',
#    'L1M.RS-SINR.Bin82.UEID.Cell6','L1M.RS-SINR.Bin94.UEID.Cell6','L1M.RS-SINR.Bin127.UEID.Cell6',
#    'L1M.RS-SINR.Bin34.UEID.Cell7','L1M.RS-SINR.Bin46.UEID.Cell7','L1M.RS-SINR.Bin58.UEID.Cell7','L1M.RS-SINR.Bin70.UEID.Cell7',
#    'L1M.RS-SINR.Bin82.UEID.Cell7','L1M.RS-SINR.Bin94.UEID.Cell7','L1M.RS-SINR.Bin127.UEID.Cell7',
#    'L1M.RS-SINR.Bin34.UEID.Cell8','L1M.RS-SINR.Bin46.UEID.Cell8','L1M.RS-SINR.Bin58.UEID.Cell8','L1M.RS-SINR.Bin70.UEID.Cell8',
#    'L1M.RS-SINR.Bin82.UEID.Cell8','L1M.RS-SINR.Bin94.UEID.Cell8','L1M.RS-SINR.Bin127.UEID.Cell8',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell2',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell2',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell2',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell2',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell3',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell3',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell3',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell3',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell4',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell4',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell4',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell4',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell5',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell5',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell5',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell5',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell6',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell6',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell6',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell6',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell7',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell7',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell7',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell7',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell8',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell8',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell8',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell8'])
#    UE = pd.concat([UE,df])
#    #SINR
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell2','L1M.RS-SINR.Bin46.UEID.Cell2', 'L1M.RS-SINR.Bin58.UEID.Cell2',##cell2
#          'L1M.RS-SINR.Bin70.UEID.Cell2','L1M.RS-SINR.Bin82.UEID.Cell2','L1M.RS-SINR.Bin94.UEID.Cell2',
#          'L1M.RS-SINR.Bin127.UEID.Cell2']] = list(UE.loc[:,'cell_sinr2'].apply(lambda x:bin_sinr(x)))
#
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell3','L1M.RS-SINR.Bin46.UEID.Cell3','L1M.RS-SINR.Bin58.UEID.Cell3',#cell3
#          'L1M.RS-SINR.Bin70.UEID.Cell3','L1M.RS-SINR.Bin82.UEID.Cell3','L1M.RS-SINR.Bin94.UEID.Cell3',
#          'L1M.RS-SINR.Bin127.UEID.Cell3']] = list(UE.loc[:,'cell_sinr3'].apply(lambda x:bin_sinr(x)))
#
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell4','L1M.RS-SINR.Bin46.UEID.Cell4','L1M.RS-SINR.Bin58.UEID.Cell4',#cell4
#          'L1M.RS-SINR.Bin70.UEID.Cell4','L1M.RS-SINR.Bin82.UEID.Cell4','L1M.RS-SINR.Bin94.UEID.Cell4',
#          'L1M.RS-SINR.Bin127.UEID.Cell4']] = list(UE.loc[:,'cell_sinr4'].apply(lambda x:bin_sinr(x)))
#
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell5','L1M.RS-SINR.Bin46.UEID.Cell5','L1M.RS-SINR.Bin58.UEID.Cell5',#cell5
#          'L1M.RS-SINR.Bin70.UEID.Cell5','L1M.RS-SINR.Bin82.UEID.Cell5','L1M.RS-SINR.Bin94.UEID.Cell5',
#          'L1M.RS-SINR.Bin127.UEID.Cell5']] = list(UE.loc[:,'cell_sinr5'].apply(lambda x:bin_sinr(x)))
#
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell6','L1M.RS-SINR.Bin46.UEID.Cell6','L1M.RS-SINR.Bin58.UEID.Cell6',#cell6
#          'L1M.RS-SINR.Bin70.UEID.Cell6','L1M.RS-SINR.Bin82.UEID.Cell6','L1M.RS-SINR.Bin94.UEID.Cell6',
#          'L1M.RS-SINR.Bin127.UEID.Cell6']] = list(UE.loc[:,'cell_sinr6'].apply(lambda x:bin_sinr(x)))
#
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell7','L1M.RS-SINR.Bin46.UEID.Cell7','L1M.RS-SINR.Bin58.UEID.Cell7',#cell7
#          'L1M.RS-SINR.Bin70.UEID.Cell7','L1M.RS-SINR.Bin82.UEID.Cell7','L1M.RS-SINR.Bin94.UEID.Cell7',
#          'L1M.RS-SINR.Bin127.UEID.Cell7']] = list(UE.loc[:,'cell_sinr7'].apply(lambda x:bin_sinr(x)))
#
#    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell8','L1M.RS-SINR.Bin46.UEID.Cell8','L1M.RS-SINR.Bin58.UEID.Cell8',#cell8
#          'L1M.RS-SINR.Bin70.UEID.Cell8','L1M.RS-SINR.Bin82.UEID.Cell8','L1M.RS-SINR.Bin94.UEID.Cell8',
#          'L1M.RS-SINR.Bin127.UEID.Cell8']] = list(UE.loc[:,'cell_sinr8'].apply(lambda x:bin_sinr(x)))
#    #mcs
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell2',#cell2
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell2',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell2']] = \
#            list(UE.loc[:,'cell_mcs2'].apply(lambda x:bin_mcs(x)))
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell3',#cell3
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell3',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell3']] = \
#            list(UE.loc[:,'cell_mcs3'].apply(lambda x:bin_mcs(x)))
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell4',#cell4
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell4',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell4']] = \
#            list(UE.loc[:,'cell_mcs4'].apply(lambda x:bin_mcs(x)))
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell5',#cell5
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell5',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell5']] = \
#            list(UE.loc[:,'cell_mcs5'].apply(lambda x:bin_mcs(x)))
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell6',#cell6
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell6',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell6']] = \
#            list(UE.loc[:,'cell_mcs6'].apply(lambda x:bin_mcs(x)))
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell7',#cell7
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell7',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell7']] = \
#            list(UE.loc[:,'cell_mcs7'].apply(lambda x:bin_mcs(x)))
#    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell8',#cell8
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell8',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell8']] = \
#            list(UE.loc[:,'cell_mcs8'].apply(lambda x:bin_mcs(x)))
#    UE.drop(['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#         'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'],axis=1,inplace=True)
#    return UE
def get_SINR_MCS_UEID(rx):
    # rx['time_cellid_imsi'] =rx.apply(lambda x: (x['Time'],x['cellId'],x['IMSI']),axis=1)
    rx_df = rx[['Time', 'IMSI', 'time_cellid_imsi', 'cellId']]
    rx_df.drop_duplicates(inplace=True)
    rx_df['sinr'] = rx_df.apply(lambda x: get_new_sinr(x, rx), axis=1)
    rx_df['mcs'] = rx_df.apply(lambda x: get_new_mcs(x, rx), axis=1)
    rx_df['cell_sinr2'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 2 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr3'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 3 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr4'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 4 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr5'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 5 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr6'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 6 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr7'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 7 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr8'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 8 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_mcs2'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 2 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs3'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 3 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs4'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 4 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs5'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 5 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs6'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 6 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs7'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 7 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs8'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 8 else np.array([], dtype='float64'), axis=1)

    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell2', 'L1M.RS-SINR.Bin46.UEID.Cell2', 'L1M.RS-SINR.Bin58.UEID.Cell2',  ##cell2
     'L1M.RS-SINR.Bin70.UEID.Cell2', 'L1M.RS-SINR.Bin82.UEID.Cell2', 'L1M.RS-SINR.Bin94.UEID.Cell2',
     'L1M.RS-SINR.Bin127.UEID.Cell2']] = list(rx_df.loc[:, 'cell_sinr2'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell3', 'L1M.RS-SINR.Bin46.UEID.Cell3', 'L1M.RS-SINR.Bin58.UEID.Cell3',  # cell3
     'L1M.RS-SINR.Bin70.UEID.Cell3', 'L1M.RS-SINR.Bin82.UEID.Cell3', 'L1M.RS-SINR.Bin94.UEID.Cell3',
     'L1M.RS-SINR.Bin127.UEID.Cell3']] = list(rx_df.loc[:, 'cell_sinr3'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell4', 'L1M.RS-SINR.Bin46.UEID.Cell4', 'L1M.RS-SINR.Bin58.UEID.Cell4',  # cell4
     'L1M.RS-SINR.Bin70.UEID.Cell4', 'L1M.RS-SINR.Bin82.UEID.Cell4', 'L1M.RS-SINR.Bin94.UEID.Cell4',
     'L1M.RS-SINR.Bin127.UEID.Cell4']] = list(rx_df.loc[:, 'cell_sinr4'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell5', 'L1M.RS-SINR.Bin46.UEID.Cell5', 'L1M.RS-SINR.Bin58.UEID.Cell5',  # cell5
     'L1M.RS-SINR.Bin70.UEID.Cell5', 'L1M.RS-SINR.Bin82.UEID.Cell5', 'L1M.RS-SINR.Bin94.UEID.Cell5',
     'L1M.RS-SINR.Bin127.UEID.Cell5']] = list(rx_df.loc[:, 'cell_sinr5'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell6', 'L1M.RS-SINR.Bin46.UEID.Cell6', 'L1M.RS-SINR.Bin58.UEID.Cell6',  # cell6
     'L1M.RS-SINR.Bin70.UEID.Cell6', 'L1M.RS-SINR.Bin82.UEID.Cell6', 'L1M.RS-SINR.Bin94.UEID.Cell6',
     'L1M.RS-SINR.Bin127.UEID.Cell6']] = list(rx_df.loc[:, 'cell_sinr6'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell7', 'L1M.RS-SINR.Bin46.UEID.Cell7', 'L1M.RS-SINR.Bin58.UEID.Cell7',  # cell7
     'L1M.RS-SINR.Bin70.UEID.Cell7', 'L1M.RS-SINR.Bin82.UEID.Cell7', 'L1M.RS-SINR.Bin94.UEID.Cell7',
     'L1M.RS-SINR.Bin127.UEID.Cell7']] = list(rx_df.loc[:, 'cell_sinr7'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['L1M.RS-SINR.Bin34.UEID.Cell8', 'L1M.RS-SINR.Bin46.UEID.Cell8', 'L1M.RS-SINR.Bin58.UEID.Cell8',  # cell8
     'L1M.RS-SINR.Bin70.UEID.Cell8', 'L1M.RS-SINR.Bin82.UEID.Cell8', 'L1M.RS-SINR.Bin94.UEID.Cell8',
     'L1M.RS-SINR.Bin127.UEID.Cell8']] = list(rx_df.loc[:, 'cell_sinr8'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell2', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell2',  # cell2
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell2', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell2',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell2', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell2']] = \
        list(rx_df.loc[:, 'cell_mcs2'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell3', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell3',  # cell3
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell3', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell3',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell3', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell3']] = \
        list(rx_df.loc[:, 'cell_mcs3'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell4', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell4',  # cell4
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell4', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell4',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell4', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell4']] = \
        list(rx_df.loc[:, 'cell_mcs4'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell5', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell5',  # cell5
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell5', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell5',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell5', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell5']] = \
        list(rx_df.loc[:, 'cell_mcs5'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell6', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell6',  # cell6
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell6', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell6',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell6', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell6']] = \
        list(rx_df.loc[:, 'cell_mcs6'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell7', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell7',  # cell7
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell7', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell7',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell7', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell7']] = \
        list(rx_df.loc[:, 'cell_mcs7'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:,
    ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell8', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell8',  # cell8
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell8', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell8',
     'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell8', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell8']] = \
        list(rx_df.loc[:, 'cell_mcs8'].apply(lambda x: bin_mcs(x)))

    rx_df.drop(['time_cellid_imsi', 'mcs', 'sinr', 'cellId', 'cell_sinr2', 'cell_sinr3', 'cell_sinr4', 'cell_sinr5',
                'cell_sinr6',
                'cell_sinr7', 'cell_sinr8', 'cell_mcs2', 'cell_mcs3', 'cell_mcs4', 'cell_mcs5', 'cell_mcs6',
                'cell_mcs7', 'cell_mcs8'], axis=1, inplace=True)
    rx_df.loc[:, :] = list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def get_new_sinr(row, df):
    return df[df['time_cellid_imsi'] == row['time_cellid_imsi']]['SINR(dB)'].values


def get_new_mcs(row, df):
    return df[df['time_cellid_imsi'] == row['time_cellid_imsi']]['mcs'].values


def get_new_sinr_cell(row, df):
    return df[df['time_cellid'] == row['time_cellid']]['SINR(dB)'].values


def get_new_mcs_cell(row, df):
    return df[df['time_cellid'] == row['time_cellid']]['mcs'].values


def TBTotNbrDl1UEID(rx_DL):
    rx_df = rx_DL.groupby(['cellId', 'IMSI', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(
            columns=['Time', 'IMSI', 'TB.TotNbrDl.1.UEID.Cell2', 'TB.TotNbrDl.1.UEID.Cell3', 'TB.TotNbrDl.1.UEID.Cell4',
                     'TB.TotNbrDl.1.UEID.Cell5', 'TB.TotNbrDl.1.UEID.Cell6', 'TB.TotNbrDl.1.UEID.Cell7',
                     'TB.TotNbrDl.1.UEID.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDl.1.UEID.Cell2', 'TB.TotNbrDl.1.UEID.Cell3', 'TB.TotNbrDl.1.UEID.Cell4',
                  'TB.TotNbrDl.1.UEID.Cell5', 'TB.TotNbrDl.1.UEID.Cell6', 'TB.TotNbrDl.1.UEID.Cell7',
                  'TB.TotNbrDl.1.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'TB.TotNbrDl.1.UEID.Cell2', 'TB.TotNbrDl.1.UEID.Cell3', 'TB.TotNbrDl.1.UEID.Cell4',
                   'TB.TotNbrDl.1.UEID.Cell5', 'TB.TotNbrDl.1.UEID.Cell6', 'TB.TotNbrDl.1.UEID.Cell7',
                   'TB.TotNbrDl.1.UEID.Cell8']]

    rx_df.loc[:, ['Time', 'IMSI', 'TB.TotNbrDl.1.UEID.Cell2', 'TB.TotNbrDl.1.UEID.Cell3', 'TB.TotNbrDl.1.UEID.Cell4',
                  'TB.TotNbrDl.1.UEID.Cell5', 'TB.TotNbrDl.1.UEID.Cell6', 'TB.TotNbrDl.1.UEID.Cell7',
                  'TB.TotNbrDl.1.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitialUEID(rx_DL):
    rx_df = rx_DL[rx_DL['rv'] == 0].groupby(['cellId', 'IMSI', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'IMSI', 'TB.TotNbrDlInitial.UEID.Cell2', 'TB.TotNbrDlInitial.UEID.Cell3',
                                     'TB.TotNbrDlInitial.UEID.Cell4', 'TB.TotNbrDlInitial.UEID.Cell5',
                                     'TB.TotNbrDlInitial.UEID.Cell6', 'TB.TotNbrDlInitial.UEID.Cell7',
                                     'TB.TotNbrDlInitial.UEID.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDlInitial.UEID.Cell2', 'TB.TotNbrDlInitial.UEID.Cell3',
                  'TB.TotNbrDlInitial.UEID.Cell4', 'TB.TotNbrDlInitial.UEID.Cell5',
                  'TB.TotNbrDlInitial.UEID.Cell6', 'TB.TotNbrDlInitial.UEID.Cell7',
                  'TB.TotNbrDlInitial.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))
    rx_df = rx_df[['Time', 'IMSI', 'TB.TotNbrDlInitial.UEID.Cell2', 'TB.TotNbrDlInitial.UEID.Cell3',
                   'TB.TotNbrDlInitial.UEID.Cell4', 'TB.TotNbrDlInitial.UEID.Cell5',
                   'TB.TotNbrDlInitial.UEID.Cell6', 'TB.TotNbrDlInitial.UEID.Cell7',
                   'TB.TotNbrDlInitial.UEID.Cell8']]
    rx_df.loc[:, ['Time', 'IMSI', 'TB.TotNbrDlInitial.UEID.Cell2', 'TB.TotNbrDlInitial.UEID.Cell3',
                  'TB.TotNbrDlInitial.UEID.Cell4', 'TB.TotNbrDlInitial.UEID.Cell5',
                  'TB.TotNbrDlInitial.UEID.Cell6', 'TB.TotNbrDlInitial.UEID.Cell7',
                  'TB.TotNbrDlInitial.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def DRBPdcpBitrateQOSUEID(rx_DL):
    rx_df = rx_DL[rx_DL['rv'] == 0].groupby(['cellId', 'IMSI', 'Time']).sum().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'IMSI', 'DRB.PdcpBitrate.QOS.UEID.Node2', 'DRB.PdcpBitrate.QOS.UEID.Node3',
                                     'DRB.PdcpBitrate.QOS.UEID.Node4', 'DRB.PdcpBitrate.QOS.UEID.Node5',
                                     'DRB.PdcpBitrate.QOS.UEID.Node6', 'DRB.PdcpBitrate.QOS.UEID.Node7',
                                     'DRB.PdcpBitrate.QOS.UEID.Node8'])
    rx_df.loc[:, ['DRB.PdcpBitrate.QOS.UEID.Node2', 'DRB.PdcpBitrate.QOS.UEID.Node3',
                  'DRB.PdcpBitrate.QOS.UEID.Node4', 'DRB.PdcpBitrate.QOS.UEID.Node5',
                  'DRB.PdcpBitrate.QOS.UEID.Node6', 'DRB.PdcpBitrate.QOS.UEID.Node7',
                  'DRB.PdcpBitrate.QOS.UEID.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize_summation(x), axis=1))
    rx_df = rx_df[['Time', 'IMSI', 'DRB.PdcpBitrate.QOS.UEID.Node2', 'DRB.PdcpBitrate.QOS.UEID.Node3',
                   'DRB.PdcpBitrate.QOS.UEID.Node4', 'DRB.PdcpBitrate.QOS.UEID.Node5',
                   'DRB.PdcpBitrate.QOS.UEID.Node6', 'DRB.PdcpBitrate.QOS.UEID.Node7',
                   'DRB.PdcpBitrate.QOS.UEID.Node8']]
    rx_df.loc[:, ['Time', 'IMSI', 'DRB.PdcpBitrate.QOS.UEID.Node2', 'DRB.PdcpBitrate.QOS.UEID.Node3',
                  'DRB.PdcpBitrate.QOS.UEID.Node4', 'DRB.PdcpBitrate.QOS.UEID.Node5',
                  'DRB.PdcpBitrate.QOS.UEID.Node6', 'DRB.PdcpBitrate.QOS.UEID.Node7',
                  'DRB.PdcpBitrate.QOS.UEID.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBInitialErrNbrDlUEID(rx_DL):
    rx_df = rx_DL[rx_DL['rv'] != 0].groupby(['cellId', 'IMSI', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'IMSI', 'TB.InitialErrNbrDl.UEID.Cell2', 'TB.InitialErrNbrDl.UEID.Cell3',
                                     'TB.InitialErrNbrDl.UEID.Cell4', 'TB.InitialErrNbrDl.UEID.Cell5',
                                     'TB.InitialErrNbrDl.UEID.Cell6', 'TB.InitialErrNbrDl.UEID.Cell7',
                                     'TB.InitialErrNbrDl.UEID.Cell8'])

    rx_df.loc[:, ['TB.InitialErrNbrDl.UEID.Cell2', 'TB.InitialErrNbrDl.UEID.Cell3',
                  'TB.InitialErrNbrDl.UEID.Cell4', 'TB.InitialErrNbrDl.UEID.Cell5',
                  'TB.InitialErrNbrDl.UEID.Cell6', 'TB.InitialErrNbrDl.UEID.Cell7',
                  'TB.InitialErrNbrDl.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))
    rx_df = rx_df[['Time', 'IMSI', 'TB.InitialErrNbrDl.UEID.Cell2', 'TB.InitialErrNbrDl.UEID.Cell3',
                   'TB.InitialErrNbrDl.UEID.Cell4', 'TB.InitialErrNbrDl.UEID.Cell5',
                   'TB.InitialErrNbrDl.UEID.Cell6', 'TB.InitialErrNbrDl.UEID.Cell7',
                   'TB.InitialErrNbrDl.UEID.Cell8']]
    rx_df.loc[:, ['Time', 'IMSI', 'TB.InitialErrNbrDl.UEID.Cell2', 'TB.InitialErrNbrDl.UEID.Cell3',
                  'TB.InitialErrNbrDl.UEID.Cell4', 'TB.InitialErrNbrDl.UEID.Cell5',
                  'TB.InitialErrNbrDl.UEID.Cell6', 'TB.InitialErrNbrDl.UEID.Cell7',
                  'TB.InitialErrNbrDl.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitialQpskUEID(rx_DL):
    rx_df = rx_DL[rx_DL['mcs'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])].groupby(
        ['cellId', 'IMSI', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(
            columns=['Time', 'IMSI', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell2', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
                     'TB.TotNbrDlInitial.Qpsk.UEID.Cell4', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
                     'TB.TotNbrDlInitial.Qpsk.UEID.Cell6', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
                     'TB.TotNbrDlInitial.Qpsk.UEID.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDlInitial.Qpsk.UEID.Cell2', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
                  'TB.TotNbrDlInitial.Qpsk.UEID.Cell4', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
                  'TB.TotNbrDlInitial.Qpsk.UEID.Cell6', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
                  'TB.TotNbrDlInitial.Qpsk.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell2', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
                   'TB.TotNbrDlInitial.Qpsk.UEID.Cell4', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
                   'TB.TotNbrDlInitial.Qpsk.UEID.Cell6', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
                   'TB.TotNbrDlInitial.Qpsk.UEID.Cell8']]

    rx_df.loc[:, ['Time', 'IMSI', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell2', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
                  'TB.TotNbrDlInitial.Qpsk.UEID.Cell4', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
                  'TB.TotNbrDlInitial.Qpsk.UEID.Cell6', 'TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
                  'TB.TotNbrDlInitial.Qpsk.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitial16QamUEID(rx_DL):
    rx_df = rx_DL[rx_DL['mcs'].isin([10, 11, 12, 13, 14, 15, 16])].groupby(
        ['cellId', 'IMSI', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(
            columns=['Time', 'IMSI', 'TB.TotNbrDlInitial.16Qam.UEID.Cell2', 'TB.TotNbrDlInitial.16Qam.UEID.Cell3',
                     'TB.TotNbrDlInitial.16Qam.UEID.Cell4', 'TB.TotNbrDlInitial.16Qam.UEID.Cell5',
                     'TB.TotNbrDlInitial.16Qam.UEID.Cell6', 'TB.TotNbrDlInitial.16Qam.UEID.Cell7',
                     'TB.TotNbrDlInitial.16Qam.UEID.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDlInitial.16Qam.UEID.Cell2', 'TB.TotNbrDlInitial.16Qam.UEID.Cell3',
                  'TB.TotNbrDlInitial.16Qam.UEID.Cell4', 'TB.TotNbrDlInitial.16Qam.UEID.Cell5',
                  'TB.TotNbrDlInitial.16Qam.UEID.Cell6', 'TB.TotNbrDlInitial.16Qam.UEID.Cell7',
                  'TB.TotNbrDlInitial.16Qam.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'TB.TotNbrDlInitial.16Qam.UEID.Cell2', 'TB.TotNbrDlInitial.16Qam.UEID.Cell3',
                   'TB.TotNbrDlInitial.16Qam.UEID.Cell4', 'TB.TotNbrDlInitial.16Qam.UEID.Cell5',
                   'TB.TotNbrDlInitial.16Qam.UEID.Cell6', 'TB.TotNbrDlInitial.16Qam.UEID.Cell7',
                   'TB.TotNbrDlInitial.16Qam.UEID.Cell8']]

    rx_df.loc[:, ['Time', 'IMSI', 'TB.TotNbrDlInitial.16Qam.UEID.Cell2', 'TB.TotNbrDlInitial.16Qam.UEID.Cell3',
                  'TB.TotNbrDlInitial.16Qam.UEID.Cell4', 'TB.TotNbrDlInitial.16Qam.UEID.Cell5',
                  'TB.TotNbrDlInitial.16Qam.UEID.Cell6', 'TB.TotNbrDlInitial.16Qam.UEID.Cell7',
                  'TB.TotNbrDlInitial.16Qam.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitial64QamUEID(rx_DL):
    rx_df = rx_DL[rx_DL['mcs'].isin([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])].groupby(
        ['cellId', 'IMSI', 'Time']).count().reset_index()

    if rx_df.empty:
        return pd.DataFrame(
            columns=['Time', 'IMSI', 'TB.TotNbrDlInitial.64Qam.UEID.Cell2', 'TB.TotNbrDlInitial.64Qam.UEID.Cell3',
                     'TB.TotNbrDlInitial.64Qam.UEID.Cell4', 'TB.TotNbrDlInitial.64Qam.UEID.Cell5',
                     'TB.TotNbrDlInitial.64Qam.UEID.Cell6', 'TB.TotNbrDlInitial.64Qam.UEID.Cell7',
                     'TB.TotNbrDlInitial.64Qam.UEID.Cell8'])
    rx_df.loc[:, ['TB.TotNbrDlInitial.64Qam.UEID.Cell2', 'TB.TotNbrDlInitial.64Qam.UEID.Cell3',
                  'TB.TotNbrDlInitial.64Qam.UEID.Cell4', 'TB.TotNbrDlInitial.64Qam.UEID.Cell5',
                  'TB.TotNbrDlInitial.64Qam.UEID.Cell6', 'TB.TotNbrDlInitial.64Qam.UEID.Cell7',
                  'TB.TotNbrDlInitial.64Qam.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'TB.TotNbrDlInitial.64Qam.UEID.Cell2', 'TB.TotNbrDlInitial.64Qam.UEID.Cell3',
                   'TB.TotNbrDlInitial.64Qam.UEID.Cell4', 'TB.TotNbrDlInitial.64Qam.UEID.Cell5',
                   'TB.TotNbrDlInitial.64Qam.UEID.Cell6', 'TB.TotNbrDlInitial.64Qam.UEID.Cell7',
                   'TB.TotNbrDlInitial.64Qam.UEID.Cell8']]

    rx_df.loc[:, ['Time', 'IMSI', 'TB.TotNbrDlInitial.64Qam.UEID.Cell2', 'TB.TotNbrDlInitial.64Qam.UEID.Cell3',
                  'TB.TotNbrDlInitial.64Qam.UEID.Cell4', 'TB.TotNbrDlInitial.64Qam.UEID.Cell5',
                  'TB.TotNbrDlInitial.64Qam.UEID.Cell6', 'TB.TotNbrDlInitial.64Qam.UEID.Cell7',
                  'TB.TotNbrDlInitial.64Qam.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def UELteUEID(dlpdcp):
    dlpdcp['DRB.PdcpSduDelayDl.UEID'] = dlpdcp['delay'] * 10000
    dlpdcp['DRB.PdcpSduBitRateDl.UEID'] = ((dlpdcp['TxBytes'] * 8) / (0.01)) / 1000
    dlpdcp['DRB.PdcpSduVolumeDl_Filter.UEID'] = (dlpdcp['TxBytes'] * 8) / (10 ** 3)
    rx_df = dlpdcp[['start', 'IMSI', 'nTxPDUs', 'DRB.PdcpSduDelayDl.UEID',
                    'DRB.PdcpSduBitRateDl.UEID', 'DRB.PdcpSduVolumeDl_Filter.UEID']]
    rx_df.rename(columns={'start': 'Time', 'nTxPDUs': 'Tot.PdcpSduNbrDl.UEID'}, inplace=True)
    dlpdcp.drop(['DRB.PdcpSduDelayDl.UEID', 'DRB.PdcpSduBitRateDl.UEID', 'DRB.PdcpSduVolumeDl_Filter.UEID'], axis=1,
                inplace=True)
    return rx_df


def DRBPdcpPduNbrDlQosUEID(DlRlcStats):
    rx_df = DlRlcStats[['start', 'IMSI', 'CellId', 'nTxPDUs']]
    rx_df.rename(columns={'start': 'Time', 'nTxPDUs': 'DRB.PdcpPduNbrDl.Qos.UEID'}, inplace=True)
    if rx_df.empty:
        return pd.DataFrame(
            columns=['Time', 'IMSI', 'DRB.PdcpPduNbrDl.Qos.UEID.Node2', 'DRB.PdcpPduNbrDl.Qos.UEID.Node3',
                     'DRB.PdcpPduNbrDl.Qos.UEID.Node4', 'DRB.PdcpPduNbrDl.Qos.UEID.Node5',
                     'DRB.PdcpPduNbrDl.Qos.UEID.Node6', 'DRB.PdcpPduNbrDl.Qos.UEID.Node7',
                     'DRB.PdcpPduNbrDl.Qos.UEID.Node8'])
    rx_df.loc[:, ['DRB.PdcpPduNbrDl.Qos.UEID.Node2', 'DRB.PdcpPduNbrDl.Qos.UEID.Node3',
                  'DRB.PdcpPduNbrDl.Qos.UEID.Node4', 'DRB.PdcpPduNbrDl.Qos.UEID.Node5',
                  'DRB.PdcpPduNbrDl.Qos.UEID.Node6', 'DRB.PdcpPduNbrDl.Qos.UEID.Node7',
                  'DRB.PdcpPduNbrDl.Qos.UEID.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: PdcpPduNbrDl_optimize(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'DRB.PdcpPduNbrDl.Qos.UEID.Node2', 'DRB.PdcpPduNbrDl.Qos.UEID.Node3',
                   'DRB.PdcpPduNbrDl.Qos.UEID.Node4', 'DRB.PdcpPduNbrDl.Qos.UEID.Node5',
                   'DRB.PdcpPduNbrDl.Qos.UEID.Node6', 'DRB.PdcpPduNbrDl.Qos.UEID.Node7',
                   'DRB.PdcpPduNbrDl.Qos.UEID.Node8']]

    rx_df.loc[:, ['Time', 'IMSI', 'DRB.PdcpPduNbrDl.Qos.UEID.Node2', 'DRB.PdcpPduNbrDl.Qos.UEID.Node3',
                  'DRB.PdcpPduNbrDl.Qos.UEID.Node4', 'DRB.PdcpPduNbrDl.Qos.UEID.Node5',
                  'DRB.PdcpPduNbrDl.Qos.UEID.Node6', 'DRB.PdcpPduNbrDl.Qos.UEID.Node7',
                  'DRB.PdcpPduNbrDl.Qos.UEID.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def DRBIPTimeDLQOSUEID(rx_DL):
    rx_df = rx_DL.groupby(['cellId', 'IMSI', 'Time']).sum().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'IMSI', 'DRB.IPTimeDL.QOS.UEID.Cell2', 'DRB.IPTimeDL.QOS.UEID.Cell3',
                                     'DRB.IPTimeDL.QOS.UEID.Cell4', 'DRB.IPTimeDL.QOS.UEID.Cell5',
                                     'DRB.IPTimeDL.QOS.UEID.Cell6', 'DRB.IPTimeDL.QOS.UEID.Cell7',
                                     'DRB.IPTimeDL.QOS.UEID.Cell8'])
    rx_df.loc[:, ['DRB.IPTimeDL.QOS.UEID.Cell2', 'DRB.IPTimeDL.QOS.UEID.Cell3',
                  'DRB.IPTimeDL.QOS.UEID.Cell4', 'DRB.IPTimeDL.QOS.UEID.Cell5',
                  'DRB.IPTimeDL.QOS.UEID.Cell6', 'DRB.IPTimeDL.QOS.UEID.Cell7',
                  'DRB.IPTimeDL.QOS.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize_IPTimeDL(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'DRB.IPTimeDL.QOS.UEID.Cell2', 'DRB.IPTimeDL.QOS.UEID.Cell3',
                   'DRB.IPTimeDL.QOS.UEID.Cell4', 'DRB.IPTimeDL.QOS.UEID.Cell5',
                   'DRB.IPTimeDL.QOS.UEID.Cell6', 'DRB.IPTimeDL.QOS.UEID.Cell7',
                   'DRB.IPTimeDL.QOS.UEID.Cell8']]

    rx_df.loc[:, ['Time', 'IMSI', 'DRB.IPTimeDL.QOS.UEID.Cell2', 'DRB.IPTimeDL.QOS.UEID.Cell3',
                  'DRB.IPTimeDL.QOS.UEID.Cell4', 'DRB.IPTimeDL.QOS.UEID.Cell5',
                  'DRB.IPTimeDL.QOS.UEID.Cell6', 'DRB.IPTimeDL.QOS.UEID.Cell7',
                  'DRB.IPTimeDL.QOS.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def QosFlowPdcpPduVolumeDL_FilterUEID(DlRlcStats):
    rx_df = DlRlcStats[['start', 'IMSI', 'CellId', 'TxBytes']]
    rx_df['TxBytes'] = (rx_df['TxBytes'] * 8) / 10 ** 3
    rx_df.rename(columns={'start': 'Time', 'TxBytes': 'QosFlow.PdcpPduVolumeDL_Filter.UEID'}, inplace=True)
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'IMSI', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2',
                                     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
                                     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4',
                                     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
                                     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6',
                                     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
                                     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8'])

    rx_df.loc[:, ['QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
                  'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
                  'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
                  'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize_PdcpPduVolumeDL_Filter(x), axis=1))

    rx_df = rx_df[
        ['Time', 'IMSI', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
         'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
         'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
         'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8']]

    rx_df.loc[:,
    ['Time', 'IMSI', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
     'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def RRUPrbUsedDlUEID(RxPacketTrace):
    # Periodicity = 10*4
    DR = 10 * 4 * 14
    rx_df = RxPacketTrace.groupby(['cellId', 'IMSI', 'Time']).sum().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'IMSI', 'RRU.PrbUsedDl.UEID.Cell2', 'RRU.PrbUsedDl.UEID.Cell3',
                                     'RRU.PrbUsedDl.UEID.Cell4', 'RRU.PrbUsedDl.UEID.Cell5',
                                     'RRU.PrbUsedDl.UEID.Cell6', 'RRU.PrbUsedDl.UEID.Cell7',
                                     'RRU.PrbUsedDl.UEID.Cell8'])

    rx_df.loc[:, ['RRU.PrbUsedDl.UEID.Cell2', 'RRU.PrbUsedDl.UEID.Cell3',
                  'RRU.PrbUsedDl.UEID.Cell4', 'RRU.PrbUsedDl.UEID.Cell5',
                  'RRU.PrbUsedDl.UEID.Cell6', 'RRU.PrbUsedDl.UEID.Cell7',
                  'RRU.PrbUsedDl.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize_UE_sp_PRB_schedule(x), axis=1))

    rx_df = rx_df[['Time', 'IMSI', 'RRU.PrbUsedDl.UEID.Cell2', 'RRU.PrbUsedDl.UEID.Cell3',
                   'RRU.PrbUsedDl.UEID.Cell4', 'RRU.PrbUsedDl.UEID.Cell5',
                   'RRU.PrbUsedDl.UEID.Cell6', 'RRU.PrbUsedDl.UEID.Cell7',
                   'RRU.PrbUsedDl.UEID.Cell8']]

    rx_df.loc[:, ['Time', 'IMSI', 'RRU.PrbUsedDl.UEID.Cell2', 'RRU.PrbUsedDl.UEID.Cell3',
                  'RRU.PrbUsedDl.UEID.Cell4', 'RRU.PrbUsedDl.UEID.Cell5',
                  'RRU.PrbUsedDl.UEID.Cell6', 'RRU.PrbUsedDl.UEID.Cell7',
                  'RRU.PrbUsedDl.UEID.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


# Cell level functions

def initiate_CELL():
    time_window = 0.01
    times = np.arange(0.01, 6.001, time_window)
    CELL = pd.DataFrame({'Time': times})
    CELL['Time'] = CELL['Time'].apply(lambda x: round(x, 4))
    return CELL


def initial_rx_sinr_mcs_cell(RxPacketTrace):
    rx_time = RxPacketTrace['Time'].unique()
    rx_sinr_mcs_df = pd.DataFrame({'Time': rx_time})
    rx_sinr_mcs_df['Time'] = rx_sinr_mcs_df['Time'].apply(lambda x: round(x, 4))
    return rx_sinr_mcs_df


# def get_SINR_MCS_CELL(CELL,rx_sinr_mcs_df):
#
#    df = pd.DataFrame(columns = ['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#                             'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'])
#    rx_sinr_mcs_df = pd.concat([rx_sinr_mcs_df,df])
#
#    #populate arrays as it's easy for later use
#    x = [np.array([],dtype='float64')]*len(rx_sinr_mcs_df)
#    rx_sinr_mcs_df[['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#                    'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8']] = \
#                    x,x,x,x,x,x,x,x,x,x,x,x,x,x
#    rx_sinr_mcs_df.loc[:,['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#                      'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8' ]] = \
#                list(rx_sinr_mcs_df.loc[:,['Time']].apply(lambda x:optimizer_sinr_mcs_cell(x),axis=1))
#    CELL = pd.merge(CELL,rx_sinr_mcs_df,on=['Time'],how='left')
#
#    #replace nan with empty array
#    for i in CELL.columns:
#        CELL[i].loc[CELL[i].isnull()] = [np.array([],dtype='float64')]*len(CELL[i].loc[CELL[i].isnull()])
#
#    df = pd.DataFrame(columns = ['L1M.RS-SINR.Bin34.Cell2','L1M.RS-SINR.Bin46.Cell2','L1M.RS-SINR.Bin58.Cell2','L1M.RS-SINR.Bin70.Cell2',
#    'L1M.RS-SINR.Bin82.Cell2','L1M.RS-SINR.Bin94.Cell2','L1M.RS-SINR.Bin127.Cell2',
#    'L1M.RS-SINR.Bin34.Cell3','L1M.RS-SINR.Bin46.Cell3','L1M.RS-SINR.Bin58.Cell3','L1M.RS-SINR.Bin70.Cell3',
#    'L1M.RS-SINR.Bin82.Cell3','L1M.RS-SINR.Bin94.Cell3','L1M.RS-SINR.Bin127.Cell3',
#    'L1M.RS-SINR.Bin34.Cell4','L1M.RS-SINR.Bin46.Cell4','L1M.RS-SINR.Bin58.Cell4','L1M.RS-SINR.Bin70.Cell4',
#    'L1M.RS-SINR.Bin82.Cell4','L1M.RS-SINR.Bin94.Cell4','L1M.RS-SINR.Bin127.Cell4',
#    'L1M.RS-SINR.Bin34.Cell5','L1M.RS-SINR.Bin46.Cell5','L1M.RS-SINR.Bin58.Cell5','L1M.RS-SINR.Bin70.Cell5',
#    'L1M.RS-SINR.Bin82.Cell5','L1M.RS-SINR.Bin94.Cell5','L1M.RS-SINR.Bin127.Cell5',
#    'L1M.RS-SINR.Bin34.Cell6','L1M.RS-SINR.Bin46.Cell6','L1M.RS-SINR.Bin58.Cell6','L1M.RS-SINR.Bin70.Cell6',
#    'L1M.RS-SINR.Bin82.Cell6','L1M.RS-SINR.Bin94.Cell6','L1M.RS-SINR.Bin127.Cell6',
#    'L1M.RS-SINR.Bin34.Cell7','L1M.RS-SINR.Bin46.Cell7','L1M.RS-SINR.Bin58.Cell7','L1M.RS-SINR.Bin70.Cell7',
#    'L1M.RS-SINR.Bin82.Cell7','L1M.RS-SINR.Bin94.Cell7','L1M.RS-SINR.Bin127.Cell7',
#    'L1M.RS-SINR.Bin34.Cell8','L1M.RS-SINR.Bin46.Cell8','L1M.RS-SINR.Bin58.Cell8','L1M.RS-SINR.Bin70.Cell8',
#    'L1M.RS-SINR.Bin82.Cell8','L1M.RS-SINR.Bin94.Cell8','L1M.RS-SINR.Bin127.Cell8',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell2',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell2',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell2',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell2',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell3',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell3',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell3',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell3',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell4',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell4',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell4',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell4',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell5',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell5',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell5',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell5',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell6',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell6',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell6',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell6',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell7',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell7',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell7',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell7',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell8',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell8',
#    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell8',
#                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell8'])
#    CELL = pd.concat([CELL,df])
#    #SINR
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell2','L1M.RS-SINR.Bin46.Cell2', 'L1M.RS-SINR.Bin58.Cell2',##cell2
#          'L1M.RS-SINR.Bin70.Cell2','L1M.RS-SINR.Bin82.Cell2','L1M.RS-SINR.Bin94.Cell2',
#          'L1M.RS-SINR.Bin127.Cell2']] = list(CELL.loc[:,'cell_sinr2'].apply(lambda x:bin_sinr(x)))
#
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell3','L1M.RS-SINR.Bin46.Cell3','L1M.RS-SINR.Bin58.Cell3',#cell3
#          'L1M.RS-SINR.Bin70.Cell3','L1M.RS-SINR.Bin82.Cell3','L1M.RS-SINR.Bin94.Cell3',
#          'L1M.RS-SINR.Bin127.Cell3']] = list(CELL.loc[:,'cell_sinr3'].apply(lambda x:bin_sinr(x)))
#
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell4','L1M.RS-SINR.Bin46.Cell4','L1M.RS-SINR.Bin58.Cell4',#cell4
#          'L1M.RS-SINR.Bin70.Cell4','L1M.RS-SINR.Bin82.Cell4','L1M.RS-SINR.Bin94.Cell4',
#          'L1M.RS-SINR.Bin127.Cell4']] = list(CELL.loc[:,'cell_sinr4'].apply(lambda x:bin_sinr(x)))
#
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell5','L1M.RS-SINR.Bin46.Cell5','L1M.RS-SINR.Bin58.Cell5',#cell5
#          'L1M.RS-SINR.Bin70.Cell5','L1M.RS-SINR.Bin82.Cell5','L1M.RS-SINR.Bin94.Cell5',
#          'L1M.RS-SINR.Bin127.Cell5']] = list(CELL.loc[:,'cell_sinr5'].apply(lambda x:bin_sinr(x)))
#
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell6','L1M.RS-SINR.Bin46.Cell6','L1M.RS-SINR.Bin58.Cell6',#cell6
#          'L1M.RS-SINR.Bin70.Cell6','L1M.RS-SINR.Bin82.Cell6','L1M.RS-SINR.Bin94.Cell6',
#          'L1M.RS-SINR.Bin127.Cell6']] = list(CELL.loc[:,'cell_sinr6'].apply(lambda x:bin_sinr(x)))
#
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell7','L1M.RS-SINR.Bin46.Cell7','L1M.RS-SINR.Bin58.Cell7',#cell7
#          'L1M.RS-SINR.Bin70.Cell7','L1M.RS-SINR.Bin82.Cell7','L1M.RS-SINR.Bin94.Cell7',
#          'L1M.RS-SINR.Bin127.Cell7']] = list(CELL.loc[:,'cell_sinr7'].apply(lambda x:bin_sinr(x)))
#
#    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell8','L1M.RS-SINR.Bin46.Cell8','L1M.RS-SINR.Bin58.Cell8',#cell8
#          'L1M.RS-SINR.Bin70.Cell8','L1M.RS-SINR.Bin82.Cell8','L1M.RS-SINR.Bin94.Cell8',
#          'L1M.RS-SINR.Bin127.Cell8']] = list(CELL.loc[:,'cell_sinr8'].apply(lambda x:bin_sinr(x)))
#    #mcs
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell2',#cell2
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell2',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell2']] = \
#            list(CELL.loc[:,'cell_mcs2'].apply(lambda x:bin_mcs(x)))
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell3',#cell3
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell3',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell3']] = \
#            list(CELL.loc[:,'cell_mcs3'].apply(lambda x:bin_mcs(x)))
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell4',#cell4
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell4',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell4']] = \
#            list(CELL.loc[:,'cell_mcs4'].apply(lambda x:bin_mcs(x)))
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell5',#cell5
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell5',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell5']] = \
#            list(CELL.loc[:,'cell_mcs5'].apply(lambda x:bin_mcs(x)))
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell6',#cell6
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell6',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell6']] = \
#            list(CELL.loc[:,'cell_mcs6'].apply(lambda x:bin_mcs(x)))
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell7',#cell7
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell7',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell7']] = \
#            list(CELL.loc[:,'cell_mcs7'].apply(lambda x:bin_mcs(x)))
#    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell8',#cell8
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell8',
#          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell8']] = \
#            list(CELL.loc[:,'cell_mcs8'].apply(lambda x:bin_mcs(x)))
#    CELL.drop(['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
#         'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'],axis=1,inplace=True)
#    return CELL

def get_SINR_MCS_CELL(rx):
    rx_df = rx[['Time', 'time_cellid', 'cellId']]
    rx_df.drop_duplicates(inplace=True)
    rx_df['sinr'] = rx_df.apply(lambda x: get_new_sinr_cell(x, rx), axis=1)
    rx_df['mcs'] = rx_df.apply(lambda x: get_new_mcs_cell(x, rx), axis=1)
    rx_df['cell_sinr2'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 2 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr3'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 3 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr4'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 4 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr5'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 5 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr6'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 6 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr7'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 7 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_sinr8'] = rx_df.apply(lambda x: x['sinr'] if x['cellId'] == 8 else np.array([], dtype='float64'),
                                      axis=1)
    rx_df['cell_mcs2'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 2 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs3'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 3 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs4'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 4 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs5'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 5 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs6'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 6 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs7'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 7 else np.array([], dtype='float64'), axis=1)
    rx_df['cell_mcs8'] = rx_df.apply(lambda x: x['mcs'] if x['cellId'] == 8 else np.array([], dtype='float64'), axis=1)

    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell2', 'L1M.RS-SINR.Bin46.Cell2', 'L1M.RS-SINR.Bin58.Cell2',  ##cell2
                  'L1M.RS-SINR.Bin70.Cell2', 'L1M.RS-SINR.Bin82.Cell2', 'L1M.RS-SINR.Bin94.Cell2',
                  'L1M.RS-SINR.Bin127.Cell2']] = list(rx_df.loc[:, 'cell_sinr2'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell3', 'L1M.RS-SINR.Bin46.Cell3', 'L1M.RS-SINR.Bin58.Cell3',  # cell3
                  'L1M.RS-SINR.Bin70.Cell3', 'L1M.RS-SINR.Bin82.Cell3', 'L1M.RS-SINR.Bin94.Cell3',
                  'L1M.RS-SINR.Bin127.Cell3']] = list(rx_df.loc[:, 'cell_sinr3'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell4', 'L1M.RS-SINR.Bin46.Cell4', 'L1M.RS-SINR.Bin58.Cell4',  # cell4
                  'L1M.RS-SINR.Bin70.Cell4', 'L1M.RS-SINR.Bin82.Cell4', 'L1M.RS-SINR.Bin94.Cell4',
                  'L1M.RS-SINR.Bin127.Cell4']] = list(rx_df.loc[:, 'cell_sinr4'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell5', 'L1M.RS-SINR.Bin46.Cell5', 'L1M.RS-SINR.Bin58.Cell5',  # cell5
                  'L1M.RS-SINR.Bin70.Cell5', 'L1M.RS-SINR.Bin82.Cell5', 'L1M.RS-SINR.Bin94.Cell5',
                  'L1M.RS-SINR.Bin127.Cell5']] = list(rx_df.loc[:, 'cell_sinr5'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell6', 'L1M.RS-SINR.Bin46.Cell6', 'L1M.RS-SINR.Bin58.Cell6',  # cell6
                  'L1M.RS-SINR.Bin70.Cell6', 'L1M.RS-SINR.Bin82.Cell6', 'L1M.RS-SINR.Bin94.Cell6',
                  'L1M.RS-SINR.Bin127.Cell6']] = list(rx_df.loc[:, 'cell_sinr6'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell7', 'L1M.RS-SINR.Bin46.Cell7', 'L1M.RS-SINR.Bin58.Cell7',  # cell7
                  'L1M.RS-SINR.Bin70.Cell7', 'L1M.RS-SINR.Bin82.Cell7', 'L1M.RS-SINR.Bin94.Cell7',
                  'L1M.RS-SINR.Bin127.Cell7']] = list(rx_df.loc[:, 'cell_sinr7'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['L1M.RS-SINR.Bin34.Cell8', 'L1M.RS-SINR.Bin46.Cell8', 'L1M.RS-SINR.Bin58.Cell8',  # cell8
                  'L1M.RS-SINR.Bin70.Cell8', 'L1M.RS-SINR.Bin82.Cell8', 'L1M.RS-SINR.Bin94.Cell8',
                  'L1M.RS-SINR.Bin127.Cell8']] = list(rx_df.loc[:, 'cell_sinr8'].apply(lambda x: bin_sinr(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell2', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell2',  # cell2
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell2', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell2',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell2', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell2']] = \
        list(rx_df.loc[:, 'cell_mcs2'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell3', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell3',  # cell3
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell3', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell3',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell3', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell3']] = \
        list(rx_df.loc[:, 'cell_mcs3'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell4', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell4',  # cell4
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell4', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell4',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell4', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell4']] = \
        list(rx_df.loc[:, 'cell_mcs4'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell5', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell5',  # cell5
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell5', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell5',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell5', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell5']] = \
        list(rx_df.loc[:, 'cell_mcs5'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell6', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell6',  # cell6
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell6', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell6',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell6', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell6']] = \
        list(rx_df.loc[:, 'cell_mcs6'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell7', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell7',  # cell7
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell7', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell7',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell7', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell7']] = \
        list(rx_df.loc[:, 'cell_mcs7'].apply(lambda x: bin_mcs(x)))
    rx_df.loc[:, ['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell8', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell8',  # cell8
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell8', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell8',
                  'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell8', 'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell8']] = \
        list(rx_df.loc[:, 'cell_mcs8'].apply(lambda x: bin_mcs(x)))

    rx_df.drop(
        ['time_cellid', 'mcs', 'sinr', 'cellId', 'cell_sinr2', 'cell_sinr3', 'cell_sinr4', 'cell_sinr5', 'cell_sinr6',
         'cell_sinr7', 'cell_sinr8', 'cell_mcs2', 'cell_mcs3', 'cell_mcs4', 'cell_mcs5', 'cell_mcs6',
         'cell_mcs7', 'cell_mcs8'], axis=1, inplace=True)
    rx_df.loc[:, :] = list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBInitialErrNbrDl(rx_DL):
    rx_df = rx_DL[rx_DL['rv'] != 0].groupby(['cellId', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(['Time', 'TB.InitialErrNbrDl.Cell2', 'TB.InitialErrNbrDl.Cell3',
                             'TB.InitialErrNbrDl.Cell4', 'TB.InitialErrNbrDl.Cell5',
                             'TB.InitialErrNbrDl.Cell6', 'TB.InitialErrNbrDl.Cell7',
                             'TB.InitialErrNbrDl.Cell8'])
    rx_df.loc[:, ['TB.InitialErrNbrDl.Cell2', 'TB.InitialErrNbrDl.Cell3',
                  'TB.InitialErrNbrDl.Cell4', 'TB.InitialErrNbrDl.Cell5',
                  'TB.InitialErrNbrDl.Cell6', 'TB.InitialErrNbrDl.Cell7',
                  'TB.InitialErrNbrDl.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))
    rx_df = rx_df[['Time', 'TB.InitialErrNbrDl.Cell2', 'TB.InitialErrNbrDl.Cell3',
                   'TB.InitialErrNbrDl.Cell4', 'TB.InitialErrNbrDl.Cell5',
                   'TB.InitialErrNbrDl.Cell6', 'TB.InitialErrNbrDl.Cell7',
                   'TB.InitialErrNbrDl.Cell8']]
    rx_df.loc[:, ['Time', 'TB.InitialErrNbrDl.Cell2', 'TB.InitialErrNbrDl.Cell3',
                  'TB.InitialErrNbrDl.Cell4', 'TB.InitialErrNbrDl.Cell5',
                  'TB.InitialErrNbrDl.Cell6', 'TB.InitialErrNbrDl.Cell7',
                  'TB.InitialErrNbrDl.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDl1(rx_DL):
    rx_df = rx_DL.groupby(['cellId', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'TB.TotNbrDl.1.Cell2', 'TB.TotNbrDl.1.Cell3', 'TB.TotNbrDl.1.Cell4',
                                     'TB.TotNbrDl.1.Cell5', 'TB.TotNbrDl.1.Cell6', 'TB.TotNbrDl.1.Cell7',
                                     'TB.TotNbrDl.1.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDl.1.Cell2', 'TB.TotNbrDl.1.Cell3', 'TB.TotNbrDl.1.Cell4',
                  'TB.TotNbrDl.1.Cell5', 'TB.TotNbrDl.1.Cell6', 'TB.TotNbrDl.1.Cell7', 'TB.TotNbrDl.1.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'TB.TotNbrDl.1.Cell2', 'TB.TotNbrDl.1.Cell3', 'TB.TotNbrDl.1.Cell4',
                   'TB.TotNbrDl.1.Cell5', 'TB.TotNbrDl.1.Cell6', 'TB.TotNbrDl.1.Cell7', 'TB.TotNbrDl.1.Cell8']]

    rx_df.loc[:, ['Time', 'TB.TotNbrDl.1.Cell2', 'TB.TotNbrDl.1.Cell3', 'TB.TotNbrDl.1.Cell4',
                  'TB.TotNbrDl.1.Cell5', 'TB.TotNbrDl.1.Cell6', 'TB.TotNbrDl.1.Cell7', 'TB.TotNbrDl.1.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitial(rx_DL):
    rx_df = rx_DL[rx_DL['rv'] == 0].groupby(['cellId', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'TB.TotNbrDlInitial.Cell2', 'TB.TotNbrDlInitial.Cell3',
                                     'TB.TotNbrDlInitial.Cell4', 'TB.TotNbrDlInitial.Cell5',
                                     'TB.TotNbrDlInitial.Cell6', 'TB.TotNbrDlInitial.Cell7',
                                     'TB.TotNbrDlInitial.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDlInitial.Cell2', 'TB.TotNbrDlInitial.Cell3',
                  'TB.TotNbrDlInitial.Cell4', 'TB.TotNbrDlInitial.Cell5',
                  'TB.TotNbrDlInitial.Cell6', 'TB.TotNbrDlInitial.Cell7',
                  'TB.TotNbrDlInitial.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'TB.TotNbrDlInitial.Cell2', 'TB.TotNbrDlInitial.Cell3',
                   'TB.TotNbrDlInitial.Cell4', 'TB.TotNbrDlInitial.Cell5',
                   'TB.TotNbrDlInitial.Cell6', 'TB.TotNbrDlInitial.Cell7',
                   'TB.TotNbrDlInitial.Cell8']]

    rx_df.loc[:, ['Time', 'TB.TotNbrDlInitial.Cell2', 'TB.TotNbrDlInitial.Cell3',
                  'TB.TotNbrDlInitial.Cell4', 'TB.TotNbrDlInitial.Cell5',
                  'TB.TotNbrDlInitial.Cell6', 'TB.TotNbrDlInitial.Cell7',
                  'TB.TotNbrDlInitial.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))
    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitial16Qam(rx_DL):
    rx_df = rx_DL[rx_DL['mcs'].isin([10, 11, 12, 13, 14, 15, 16])].groupby(['cellId', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'TB.TotNbrDlInitial.16Qam.Cell2', 'TB.TotNbrDlInitial.16Qam.Cell3',
                                     'TB.TotNbrDlInitial.16Qam.Cell4', 'TB.TotNbrDlInitial.16Qam.Cell5',
                                     'TB.TotNbrDlInitial.16Qam.Cell6', 'TB.TotNbrDlInitial.16Qam.Cell7',
                                     'TB.TotNbrDlInitial.16Qam.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDlInitial.16Qam.Cell2', 'TB.TotNbrDlInitial.16Qam.Cell3',
                  'TB.TotNbrDlInitial.16Qam.Cell4', 'TB.TotNbrDlInitial.16Qam.Cell5',
                  'TB.TotNbrDlInitial.16Qam.Cell6', 'TB.TotNbrDlInitial.16Qam.Cell7',
                  'TB.TotNbrDlInitial.16Qam.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'TB.TotNbrDlInitial.16Qam.Cell2', 'TB.TotNbrDlInitial.16Qam.Cell3',
                   'TB.TotNbrDlInitial.16Qam.Cell4', 'TB.TotNbrDlInitial.16Qam.Cell5',
                   'TB.TotNbrDlInitial.16Qam.Cell6', 'TB.TotNbrDlInitial.16Qam.Cell7',
                   'TB.TotNbrDlInitial.16Qam.Cell8']]

    rx_df.loc[:, ['Time', 'TB.TotNbrDlInitial.16Qam.Cell2', 'TB.TotNbrDlInitial.16Qam.Cell3',
                  'TB.TotNbrDlInitial.16Qam.Cell4', 'TB.TotNbrDlInitial.16Qam.Cell5',
                  'TB.TotNbrDlInitial.16Qam.Cell6', 'TB.TotNbrDlInitial.16Qam.Cell7',
                  'TB.TotNbrDlInitial.16Qam.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitial64Qam(rx_DL):
    rx_df = rx_DL[rx_DL['mcs'].isin([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])].groupby(
        ['cellId', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'TB.TotNbrDlInitial.64Qam.Cell2', 'TB.TotNbrDlInitial.64Qam.Cell3',
                                     'TB.TotNbrDlInitial.64Qam.Cell4',
                                     'TB.TotNbrDlInitial.64Qam.Cell5', 'TB.TotNbrDlInitial.64Qam.Cell6',
                                     'TB.TotNbrDlInitial.64Qam.Cell7',
                                     'TB.TotNbrDlInitial.64Qam.Cell8'])
    rx_df.loc[:, ['TB.TotNbrDlInitial.64Qam.Cell2', 'TB.TotNbrDlInitial.64Qam.Cell3', 'TB.TotNbrDlInitial.64Qam.Cell4',
                  'TB.TotNbrDlInitial.64Qam.Cell5', 'TB.TotNbrDlInitial.64Qam.Cell6', 'TB.TotNbrDlInitial.64Qam.Cell7',
                  'TB.TotNbrDlInitial.64Qam.Cell8']] = list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[
        ['Time', 'TB.TotNbrDlInitial.64Qam.Cell2', 'TB.TotNbrDlInitial.64Qam.Cell3', 'TB.TotNbrDlInitial.64Qam.Cell4',
         'TB.TotNbrDlInitial.64Qam.Cell5', 'TB.TotNbrDlInitial.64Qam.Cell6', 'TB.TotNbrDlInitial.64Qam.Cell7',
         'TB.TotNbrDlInitial.64Qam.Cell8']]

    rx_df.loc[:,
    ['Time', 'TB.TotNbrDlInitial.64Qam.Cell2', 'TB.TotNbrDlInitial.64Qam.Cell3', 'TB.TotNbrDlInitial.64Qam.Cell4',
     'TB.TotNbrDlInitial.64Qam.Cell5', 'TB.TotNbrDlInitial.64Qam.Cell6', 'TB.TotNbrDlInitial.64Qam.Cell7',
     'TB.TotNbrDlInitial.64Qam.Cell8']] = list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def TBTotNbrDlInitialQpsk(rx_DL):
    rx_df = rx_DL[rx_DL['mcs'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])].groupby(['cellId', 'Time']).count().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'TB.TotNbrDlInitial.Qpsk.Cell2', 'TB.TotNbrDlInitial.Qpsk.Cell3',
                                     'TB.TotNbrDlInitial.Qpsk.Cell4', 'TB.TotNbrDlInitial.Qpsk.Cell5',
                                     'TB.TotNbrDlInitial.Qpsk.Cell6', 'TB.TotNbrDlInitial.Qpsk.Cell7',
                                     'TB.TotNbrDlInitial.Qpsk.Cell8'])

    rx_df.loc[:, ['TB.TotNbrDlInitial.Qpsk.Cell2', 'TB.TotNbrDlInitial.Qpsk.Cell3',
                  'TB.TotNbrDlInitial.Qpsk.Cell4', 'TB.TotNbrDlInitial.Qpsk.Cell5',
                  'TB.TotNbrDlInitial.Qpsk.Cell6', 'TB.TotNbrDlInitial.Qpsk.Cell7',
                  'TB.TotNbrDlInitial.Qpsk.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize(x), axis=1))

    rx_df = rx_df[['Time', 'TB.TotNbrDlInitial.Qpsk.Cell2', 'TB.TotNbrDlInitial.Qpsk.Cell3',
                   'TB.TotNbrDlInitial.Qpsk.Cell4', 'TB.TotNbrDlInitial.Qpsk.Cell5',
                   'TB.TotNbrDlInitial.Qpsk.Cell6', 'TB.TotNbrDlInitial.Qpsk.Cell7',
                   'TB.TotNbrDlInitial.Qpsk.Cell8']]

    rx_df.loc[:, ['Time', 'TB.TotNbrDlInitial.Qpsk.Cell2', 'TB.TotNbrDlInitial.Qpsk.Cell3',
                  'TB.TotNbrDlInitial.Qpsk.Cell4', 'TB.TotNbrDlInitial.Qpsk.Cell5',
                  'TB.TotNbrDlInitial.Qpsk.Cell6', 'TB.TotNbrDlInitial.Qpsk.Cell7',
                  'TB.TotNbrDlInitial.Qpsk.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def RRUPrbUsedDl(RxPacketTrace):
    # Periodicity = 10*4
    # DR = Periodicity*14
    rx_df = RxPacketTrace.groupby(['cellId', 'Time']).sum().reset_index()
    if rx_df.empty:
        return pd.DataFrame(columns=['Time', 'RRU.PrbUsedDl.Cell2', 'RRU.PrbUsedDl.Cell3',
                                     'RRU.PrbUsedDl.Cell4', 'RRU.PrbUsedDl.Cell5',
                                     'RRU.PrbUsedDl.Cell6', 'RRU.PrbUsedDl.Cell7',
                                     'RRU.PrbUsedDl.Cell8'])

    rx_df.loc[:, ['RRU.PrbUsedDl.Cell2', 'RRU.PrbUsedDl.Cell3',
                  'RRU.PrbUsedDl.Cell4', 'RRU.PrbUsedDl.Cell5',
                  'RRU.PrbUsedDl.Cell6', 'RRU.PrbUsedDl.Cell7',
                  'RRU.PrbUsedDl.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize_UE_sp_PRB_schedule(x), axis=1))

    rx_df = rx_df[['Time', 'RRU.PrbUsedDl.Cell2', 'RRU.PrbUsedDl.Cell3',
                   'RRU.PrbUsedDl.Cell4', 'RRU.PrbUsedDl.Cell5',
                   'RRU.PrbUsedDl.Cell6', 'RRU.PrbUsedDl.Cell7',
                   'RRU.PrbUsedDl.Cell8']]

    rx_df.loc[:, ['Time', 'RRU.PrbUsedDl.Cell2', 'RRU.PrbUsedDl.Cell3',
                  'RRU.PrbUsedDl.Cell4', 'RRU.PrbUsedDl.Cell5',
                  'RRU.PrbUsedDl.Cell6', 'RRU.PrbUsedDl.Cell7',
                  'RRU.PrbUsedDl.Cell8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def DRBMeanActiveUeDl(RxPacketTrace, UeHandoverEndStats):
    # to be optimized
    df = pd.DataFrame(columns=['Time', 'DRB.MeanActiveUeDl.Cell2', 'DRB.MeanActiveUeDl.Cell3',
                               'DRB.MeanActiveUeDl.Cell4', 'DRB.MeanActiveUeDl.Cell5',
                               'DRB.MeanActiveUeDl.Cell6', 'DRB.MeanActiveUeDl.Cell7',
                               'DRB.MeanActiveUeDl.Cell8'])
    time_window = 0.01
    times = np.arange(0.01, 6.001, time_window)
    for time in times:
        dt = {
            'Time': round(time, 2),
            'DRB.MeanActiveUeDl.Cell2': 0, 'DRB.MeanActiveUeDl.Cell3': 0,
            'DRB.MeanActiveUeDl.Cell4': 0, 'DRB.MeanActiveUeDl.Cell5': 0,
            'DRB.MeanActiveUeDl.Cell6': 0, 'DRB.MeanActiveUeDl.Cell7': 0,
            'DRB.MeanActiveUeDl.Cell8': 0
        }
        for cell in range(2, 9):
            # operand A
            condition_A = ((RxPacketTrace['time'] >= round(time, 2))
                           & (RxPacketTrace['time'] <= round(time + 0.01, 2))
                           & (cell == RxPacketTrace['cellId']))
            # operand B
            condition_B = ((UeHandoverEndStats['Time'] >= round(time, 2))
                           & (UeHandoverEndStats['Time'] <= round(time + 0.01, 2))
                           & (cell == UeHandoverEndStats['TargetCellId']))

            result_A = set(RxPacketTrace[condition_A]['IMSI'])
            result_B = set(UeHandoverEndStats[condition_B]['IMSI'])
            result = len(result_A - result_B)
            if result != 0:
                key = f'DRB.MeanActiveUeDl.Cell{cell}'
                dt[key] = result
        df = df.append(dt, ignore_index=True)
    return df


# Node level functions

def initiate_NODE():
    time_window = 0.01
    times = np.arange(0.01, 6.001, time_window)
    NODE = pd.DataFrame({'Time': times})
    NODE['Time'] = NODE['Time'].apply(lambda x: round(x, 4))
    return NODE


def DRBPdcpSduDelayDl(DlRlcStats):
    df = pd.DataFrame(
        columns=['Time', 'DRB.PdcpSduDelayDl.Node2', 'DRB.PdcpSduDelayDl.Node3', 'DRB.PdcpSduDelayDl.Node4',
                 'DRB.PdcpSduDelayDl.Node5', 'DRB.PdcpSduDelayDl.Node6', 'DRB.PdcpSduDelayDl.Node7',
                 'DRB.PdcpSduDelayDl.Node8'])
    time_window = 0.01
    times = np.arange(0.01, 6.001, time_window)
    for time in times:
        dt = {'Time': round(time, 2), 'DRB.PdcpSduDelayDl.Node2': 0,
              'DRB.PdcpSduDelayDl.Node3': 0, 'DRB.PdcpSduDelayDl.Node4': 0, 'DRB.PdcpSduDelayDl.Node5': 0,
              'DRB.PdcpSduDelayDl.Node6': 0, 'DRB.PdcpSduDelayDl.Node7': 0, 'DRB.PdcpSduDelayDl.Node8': 0
              }
        for cell in range(2, 9):
            condition = ((round(time, 2) == DlRlcStats['start'])
                         & (cell == DlRlcStats['CellId']))
            result = 0
            if not DlRlcStats[condition].empty:
                result = (DlRlcStats[condition]['delay'] * 10000).mean()
            key = f'DRB.PdcpSduDelayDl.Node{cell}'
            dt[key] = result
        df = df.append(dt, ignore_index=True)
    return df


def DRBUEThpDl(DlRlcStats, RxPacketTrace):
    df = pd.DataFrame(columns=['Time', 'DRB.UEThpDl.Node2', 'DRB.UEThpDl.Node3',
                               'DRB.UEThpDl.Node4', 'DRB.UEThpDl.Node5', 'DRB.UEThpDl.Node6',
                               'DRB.UEThpDl.Node7', 'DRB.UEThpDl.Node8'])
    time_window = 0.01
    times = np.arange(0.01, 6.001, time_window)
    for time in times:
        dt = {
            'Time': round(time, 2),
            'DRB.UEThpDl.Node2': 0, 'DRB.UEThpDl.Node3': 0, 'DRB.UEThpDl.Node4': 0,
            'DRB.UEThpDl.Node5': 0, 'DRB.UEThpDl.Node6': 0, 'DRB.UEThpDl.Node7': 0,
            'DRB.UEThpDl.Node8': 0
        }

        for cell in range(2, 9):
            dl_condition = ((DlRlcStats['start'] == round(time, 2))
                            & (DlRlcStats['CellId'] == cell))

            rl_condition = ((RxPacketTrace['time'] >= round(time, 2))
                            & (RxPacketTrace['time'] <= round(time + 0.01, 2))
                            & (RxPacketTrace['cellId'] == cell))
            dl_res = DlRlcStats[dl_condition] * 8
            rl_res = RxPacketTrace[rl_condition]['symbol#'].sum() * 10 ** 3
            if not dl_res.empty and rl_res != 0:
                result = dl_res / rl_res
                avg = result['TxBytes'].mean()
                key = f'DRB.UEThpDl.Node{cell}'
                dt[key] = avg
        df = df.append(dt, ignore_index=True)
    return df


def QosFlowPdcpPduVolumeDL_Filter(DlRlcStats):
    rx_df = DlRlcStats[['start', 'CellId', 'TxBytes']]
    rx_df['TxBytes'] = (rx_df['TxBytes'] * 8) / 10 ** 3
    rx_df = rx_df.groupby(['start', 'CellId']).sum().reset_index()
    rx_df.rename(columns={'start': 'Time', 'TxBytes': 'QosFlow.PdcpPduVolumeDL_Filter.UEID'}, inplace=True)
    if rx_df.empty:
        return pd.DataFrame(
            columns=['Time', 'QosFlow.PdcpPduVolumeDL_Filter.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.Node3',
                     'QosFlow.PdcpPduVolumeDL_Filter.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.Node5',
                     'QosFlow.PdcpPduVolumeDL_Filter.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.Node7',
                     'QosFlow.PdcpPduVolumeDL_Filter.Node8'])

    rx_df.loc[:, ['QosFlow.PdcpPduVolumeDL_Filter.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.Node3',
                  'QosFlow.PdcpPduVolumeDL_Filter.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.Node5',
                  'QosFlow.PdcpPduVolumeDL_Filter.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.Node7',
                  'QosFlow.PdcpPduVolumeDL_Filter.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: optimize_PdcpPduVolumeDL_Filter(x), axis=1))

    rx_df = rx_df[['Time', 'QosFlow.PdcpPduVolumeDL_Filter.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.Node3',
                   'QosFlow.PdcpPduVolumeDL_Filter.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.Node5',
                   'QosFlow.PdcpPduVolumeDL_Filter.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.Node7',
                   'QosFlow.PdcpPduVolumeDL_Filter.Node8']]

    rx_df.loc[:, ['Time', 'QosFlow.PdcpPduVolumeDL_Filter.Node2', 'QosFlow.PdcpPduVolumeDL_Filter.Node3',
                  'QosFlow.PdcpPduVolumeDL_Filter.Node4', 'QosFlow.PdcpPduVolumeDL_Filter.Node5',
                  'QosFlow.PdcpPduVolumeDL_Filter.Node6', 'QosFlow.PdcpPduVolumeDL_Filter.Node7',
                  'QosFlow.PdcpPduVolumeDL_Filter.Node8']] = \
        list(rx_df.loc[:, ].apply(lambda x: max_val_cell(x, rx_df), axis=1))

    rx_df.drop_duplicates(inplace=True)
    return rx_df


def get_new_serving_cell(startStats, endStats):
    def func(row):
        condition = ((row['Time'] >= endStats['Time'])
                     & (row['IMSI'] == endStats['IMSI']))
        endStats_subset = endStats[condition]
        if endStats_subset.shape[0] != 0:
            return endStats_subset.iloc[-1, 3]
        else:
            return startStats.iloc[0, 3]

    return func


def TBTotNbrDlInitial16Qam_regression(row):
    if row['Serving_NR'] == 2:
        return row['TB.TotNbrDlInitial.16Qam.Cell2']
    elif row['Serving_NR'] == 3:
        return row['TB.TotNbrDlInitial.16Qam.Cell3']
    elif row['Serving_NR'] == 4:
        return row['TB.TotNbrDlInitial.16Qam.Cell4']
    elif row['Serving_NR'] == 5:
        return row['TB.TotNbrDlInitial.16Qam.Cell5']
    elif row['Serving_NR'] == 6:
        return row['TB.TotNbrDlInitial.16Qam.Cell6']
    elif row['Serving_NR'] == 7:
        return row['TB.TotNbrDlInitial.16Qam.Cell7']
    elif row['Serving_NR'] == 8:
        return row['TB.TotNbrDlInitial.16Qam.Cell8']


def TBTotNbrDlInitial64Qam_regression(row):
    if row['Serving_NR'] == 2:
        return row['TB.TotNbrDlInitial.64Qam.Cell2']
    elif row['Serving_NR'] == 3:
        return row['TB.TotNbrDlInitial.64Qam.Cell3']
    elif row['Serving_NR'] == 4:
        return row['TB.TotNbrDlInitial.64Qam.Cell4']
    elif row['Serving_NR'] == 5:
        return row['TB.TotNbrDlInitial.64Qam.Cell5']
    elif row['Serving_NR'] == 6:
        return row['TB.TotNbrDlInitial.64Qam.Cell6']
    elif row['Serving_NR'] == 7:
        return row['TB.TotNbrDlInitial.64Qam.Cell7']
    elif row['Serving_NR'] == 8:
        return row['TB.TotNbrDlInitial.64Qam.Cell8']


def TBTotNbrDlInitialQpsk_regression(row):
    if row['Serving_NR'] == 2:
        return row['TB.TotNbrDlInitial.Qpsk.Cell2']
    elif row['Serving_NR'] == 3:
        return row['TB.TotNbrDlInitial.Qpsk.Cell3']
    elif row['Serving_NR'] == 4:
        return row['TB.TotNbrDlInitial.Qpsk.Cell4']
    elif row['Serving_NR'] == 5:
        return row['TB.TotNbrDlInitial.Qpsk.Cell5']
    elif row['Serving_NR'] == 6:
        return row['TB.TotNbrDlInitial.Qpsk.Cell6']
    elif row['Serving_NR'] == 7:
        return row['TB.TotNbrDlInitial.Qpsk.Cell7']
    elif row['Serving_NR'] == 8:
        return row['TB.TotNbrDlInitial.Qpsk.Cell8']


def RRUPrbUsedDl_regression(row):
    if row['Serving_NR'] == 2:
        return row['RRU.PrbUsedDl.Cell2']
    elif row['Serving_NR'] == 3:
        return row['RRU.PrbUsedDl.Cell3']
    elif row['Serving_NR'] == 4:
        return row['RRU.PrbUsedDl.Cell4']
    elif row['Serving_NR'] == 5:
        return row['RRU.PrbUsedDl.Cell5']
    elif row['Serving_NR'] == 6:
        return row['RRU.PrbUsedDl.Cell6']
    elif row['Serving_NR'] == 7:
        return row['RRU.PrbUsedDl.Cell7']
    elif row['Serving_NR'] == 8:
        return row['RRU.PrbUsedDl.Cell8']


def DRBPdcpSduDelayDl_regression(row):
    if row['Serving_NR'] == 2:
        return row['DRB.PdcpSduDelayDl.Node2']
    elif row['Serving_NR'] == 3:
        return row['DRB.PdcpSduDelayDl.Node3']
    elif row['Serving_NR'] == 4:
        return row['DRB.PdcpSduDelayDl.Node4']
    elif row['Serving_NR'] == 5:
        return row['DRB.PdcpSduDelayDl.Node5']
    elif row['Serving_NR'] == 6:
        return row['DRB.PdcpSduDelayDl.Node6']
    elif row['Serving_NR'] == 7:
        return row['DRB.PdcpSduDelayDl.Node7']
    elif row['Serving_NR'] == 8:
        return row['DRB.PdcpSduDelayDl.Node8']


def QosFlowPdcpPduVolumeDL_Filter_regression(row):
    if row['Serving_NR'] == 2:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node2']
    elif row['Serving_NR'] == 3:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node3']
    elif row['Serving_NR'] == 4:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node4']
    elif row['Serving_NR'] == 5:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node5']
    elif row['Serving_NR'] == 6:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node6']
    elif row['Serving_NR'] == 7:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node7']
    elif row['Serving_NR'] == 8:
        return row['QosFlow.PdcpPduVolumeDL_Filter.Node8']


# DRB.MeanActiveUeDl
def DRBMeanActiveUeDl_regression(row):
    if row['Serving_NR'] == 2:
        return row['DRB.MeanActiveUeDl.Cell2']
    elif row['Serving_NR'] == 3:
        return row['DRB.MeanActiveUeDl.Cell3']
    elif row['Serving_NR'] == 4:
        return row['DRB.MeanActiveUeDl.Cell4']
    elif row['Serving_NR'] == 5:
        return row['DRB.MeanActiveUeDl.Cell5']
    elif row['Serving_NR'] == 6:
        return row['DRB.MeanActiveUeDl.Cell6']
    elif row['Serving_NR'] == 7:
        return row['DRB.MeanActiveUeDl.Cell7']
    elif row['Serving_NR'] == 8:
        return row['DRB.MeanActiveUeDl.Cell8']


def process_dir(directory):
    # cur_proc = mp.current_process()
    start_time = datetime.now().strftime("%H:%M:%S")
    # print(f'{cur_proc._identity}: ********* {directory} Started @ {start_time} ************')
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, directory)

    # skip dirs that were already processed
    # Add regression PM's
    if os.path.exists(os.path.join(data_dir, 'final_df.csv')):
        processed_df = pd.read_csv(os.path.join(data_dir, 'final_df.csv'))
        if 'Serving_NR' in processed_df.columns:
            del processed_df
            # print(f'{cur_proc._identity}: --------- {directory} already processed - skipping ---------')
            return
        else:
            # print(f'{cur_proc._identity}: --------- {directory} including Serving_NR ---------')
            # sort the df
            processed_df = processed_df.sort_values(['IMSI', 'Time'])
            # import handoverstartstat & endstat
            file = 'UeHandoverStartStats.txt'
            UeHandoverStartStats = pd.read_csv(os.path.join(data_dir, file), sep=' ',
                                               names=['time', 'IMSI', 'RNTI', 'SourceCellId', 'TargetCellId'])
            file = 'UeHandoverEndStats.txt'
            UeHandoverEndStats = pd.read_csv(os.path.join(data_dir, file), sep=' ',
                                             names=['time', 'IMSI', 'RNTI', 'Serving_NR'])
            # Create time into ms window
            UeHandoverEndStats['Time'] = UeHandoverEndStats['time'].apply(lambda x: float(str(x)[:4]))
            # remove duplicate rows keep thelatest record.
            UeHandoverEndStats = UeHandoverEndStats[~UeHandoverEndStats.duplicated(['IMSI', 'Time'], keep='last')]
            # Create Serving_NR
            processed_df = pd.merge(processed_df, UeHandoverEndStats[['Time', 'IMSI', 'Serving_NR']], how='left',
                                    on=['Time', 'IMSI'])

            # Forward fill
            for imsi in processed_df['IMSI'].unique():
                processed_df.loc[processed_df['IMSI'] == imsi, 'Serving_NR'] = processed_df.loc[
                    processed_df['IMSI'] == imsi, 'Serving_NR'].fillna(method='ffill')
                # extract the first record time for that IMSI HO
                time = UeHandoverEndStats[UeHandoverEndStats['IMSI'] == imsi].sort_values('time').iloc[0, :]['time']
                # extract the source cell id from uehandoverstarts for that time
                cell = UeHandoverStartStats[
                           (UeHandoverStartStats['time'] <= time) & (UeHandoverStartStats['IMSI'] == imsi)].sort_values(
                    'time').iloc[-1, :]['SourceCellId']
                processed_df.loc[processed_df['IMSI'] == imsi, 'Serving_NR'] = processed_df.loc[
                    processed_df['IMSI'] == imsi, 'Serving_NR'].fillna(cell)

            processed_df['TB.TotNbrDlInitial.16Qam'] = processed_df.apply(
                lambda x: TBTotNbrDlInitial16Qam_regression(x), axis=1)
            processed_df['TB.TotNbrDlInitial.64Qam'] = processed_df.apply(
                lambda x: TBTotNbrDlInitial64Qam_regression(x), axis=1)
            processed_df['TB.TotNbrDlInitial.Qpsk'] = processed_df.apply(lambda x: TBTotNbrDlInitialQpsk_regression(x),
                                                                         axis=1)
            processed_df['RRU.PrbUsedDl'] = processed_df.apply(lambda x: RRUPrbUsedDl_regression(x), axis=1)
            processed_df['DRB.PdcpSduDelayDl'] = processed_df.apply(lambda x: DRBPdcpSduDelayDl_regression(x), axis=1)
            processed_df['QosFlow.PdcpPduVolumeDL_Filter'] = processed_df.apply(
                lambda x: QosFlowPdcpPduVolumeDL_Filter_regression(x), axis=1)
            processed_df['DRB.MeanActiveUeDl'] = processed_df.apply(lambda x: DRBMeanActiveUeDl_regression(x), axis=1)

            # filter IMSI having more than 2 HO
            processed_df['Previous_NR'] = processed_df.groupby('IMSI').agg(HO=('Serving_NR', 'shift'))
            # Find the HO cnts
            IMSI_HO_df = processed_df[(processed_df['Serving_NR'] != processed_df['Previous_NR'])].groupby(
                ['IMSI']).count().reset_index()[['IMSI', 'Previous_NR']]
            # Filter the UE's with more than 2 HO
            IMSI_HO = IMSI_HO_df[IMSI_HO_df['Previous_NR'] >= 2]['IMSI']

            processed_df[processed_df['IMSI'].isin(IMSI_HO)][['Time', 'IMSI', 'Serving_NR',
                                                              'DRB.PdcpSduDelayDl.UEID', 'Tot.PdcpSduNbrDl.UEID',
                                                              'DRB.PdcpSduBitRateDl.UEID', 'TB.TotNbrDlInitial.16Qam',
                                                              'TB.TotNbrDlInitial.64Qam', 'TB.TotNbrDlInitial.Qpsk',
                                                              'RRU.PrbUsedDl', 'DRB.PdcpSduDelayDl',
                                                              'QosFlow.PdcpPduVolumeDL_Filter',
                                                              'DRB.MeanActiveUeDl']].to_csv(
                os.path.join(data_dir, 'Regression_PM.csv'), index=False)
            end_time = datetime.now().strftime("%H:%M:%S")
            del IMSI_HO, IMSI_HO_df, processed_df, time, cell, UeHandoverStartStats, UeHandoverEndStats
            # print(f'{cur_proc._identity}: ********* {directory} Completed @ {end_time}  ************')
            return

        # if 'new_current_serving_NR' in processed_df.columns:
        #    del processed_df
        #    print(f'{cur_proc._identity}: --------- {directory} already processed - skipping ---------')
        #    return
        # else:
        #    print(f'{cur_proc._identity}: --------- {directory} has final_df - including new column ---------')
        #    UeHandoverStartStats_ = pd.read_csv(os.path.join(data_dir, 'UeHandoverStartStats.txt'),
        #                                       sep=' ',
        #                                       names=['Time', 'IMSI', 'RNTI', 'SourceCellId', 'TargetCellId'])
        #    UeHandoverEndStats_ = pd.read_csv(os.path.join(data_dir, 'UeHandoverEndStats.txt'),
        #                                     sep=' ',
        #                                     names=['Time','IMSI','RNTI','TargetCellId'])
        #    processed_df['new_current_serving_NR'] = processed_df.agg(get_new_serving_cell(UeHandoverStartStats_, UeHandoverEndStats_), axis=1)
        #    processed_df.to_csv(os.path.join(data_dir, 'final_df.csv'), index=False)
        #    del UeHandoverStartStats_, UeHandoverEndStats_
        #    end_time = datetime.now().strftime("%H:%M:%S")
        #    print(f'{cur_proc._identity}: ********* {directory} Completed @ {end_time}  ************')
        #    return

    # select .txt files
    # file_list = [file for file in os.listdir(data_dir) if file.endswith('.txt')]
    #
    ## skip if no txt files are present
    # if len(file_list) == 0:
    #    print(f'{cur_proc._identity}: --------- {directory} does not have any text files - skipping...')
    #    return
    #
    # global DlPcdpStats, DlRlcStats, UeHandoverStartStats, UeHandoverEndStats, MmWaveSineTime, RxPacketTrace
    #
    ##create dataframes
    # for file in file_list:
    #    if file == 'DlPdcpStats.txt':
    #        DlPdcpStats = pd.read_csv(os.path.join(data_dir,file), sep='\t', index_col=False)
    #        DlPdcpStats.rename(columns={'% start': 'start'}, inplace=True)
    #        DlPdcpStats.drop(columns=['stdDev', 'min', 'max', 'stdDev.1', 'min.1', 'max.1'], inplace=True)
    #    elif file == 'DlRlcStats.txt':
    #        DlRlcStats = pd.read_csv(os.path.join(data_dir,file), sep='\t', index_col=False)
    #        DlRlcStats.rename(columns={'% start': 'start'}, inplace=True)
    #        DlRlcStats.drop(columns=['stdDev', 'min', 'max', 'stdDev.1', 'min.1', 'max.1'], inplace=True)
    #    elif file == 'UeHandoverStartStats.txt':
    #        UeHandoverStartStats = pd.read_csv(os.path.join(data_dir,file), sep=' ',
    #                        names=['Time', 'IMSI', 'RNTI', 'SourceCellId', 'TargetCellId'])
    #    elif file == 'UeHandoverEndStats.txt':
    #        UeHandoverEndStats = pd.read_csv(os.path.join(data_dir,file), sep=' ',
    #                                         names=['Time','IMSI','RNTI','TargetCellId'])
    #    elif file == 'MmWaveSinrTime.txt':
    #        MmWaveSinrTime = pd.read_csv(os.path.join(data_dir,file), sep = ' ',
    #                                     names = ['Time', 'IMSI', 'CellId', 'SINR'])
    #    elif file == 'RxPacketTrace.txt':
    #        RxPacketTrace = pd.read_csv(os.path.join(data_dir,file), sep='\t', index_col=False)
    # RxPacketTrace['IMSI'] = RxPacketTrace.agg(getIMSI, axis=1)
    ##drop nulls
    # RxPacketTrace.drop(RxPacketTrace[RxPacketTrace['IMSI'].isnull()].index, inplace=True)
    # RxPacketTrace['IMSI'] = RxPacketTrace['IMSI'].astype('int32')
    # RxPacketTrace['Time'] = RxPacketTrace['time'].apply(lambda x:float(str(x)[:4]))
    # RxPacketTrace_DL = RxPacketTrace[(RxPacketTrace['DL/UL']=='DL') & (RxPacketTrace['cellId']!=1)]
    # RxPacketTrace['time_cellid_imsi'] =RxPacketTrace.apply(lambda x: (x['Time'],x['cellId'],x['IMSI']),axis=1)
    # RxPacketTrace['time_cellid'] =RxPacketTrace.apply(lambda x: (x['Time'],x['cellId']),axis=1)
    ##***************************UE level processing*****************************************#
    ##initiate UE & rx_sinr_mcs_df
    # UE = initial_UE()
    ##rx_sinr_mcs_df = initial_rx_sinr_mcs()
    ##L1M.RS-SINR.BinX.UEID
    ##CARR.PDSCHMCSDist.BinX.BinY.BinZ.UEID
    ##UE = get_SINR_MCS_UEID(UE,rx_sinr_mcs_df)
    # UE = pd.merge(UE,get_SINR_MCS_UEID(RxPacketTrace),on=['Time','IMSI'],how='left').fillna(0)
    ##UE_sinr_mcs_bkp = UE.copy()
    ##TB.TotNbrDl.1.UEID
    # UE = pd.merge(UE,TBTotNbrDl1UEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##TB.TotNbrDlInitial.UEID
    # UE = pd.merge(UE,TBTotNbrDlInitialUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##DRB.PdcpBitrate.QOS.UEID
    ##UE = pd.merge(UE,DRBPdcpBitrateQOSUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##TB.InitialErrNbrDl.UEID
    # UE = pd.merge(UE,TBInitialErrNbrDlUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##TB.TotNbrDlInitial.Qpsk.UEID
    # UE = pd.merge(UE,TBTotNbrDlInitialQpskUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##TB.TotNbrDlInitial.16Qam.UEID
    # UE = pd.merge(UE,TBTotNbrDlInitial16QamUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##TB.TotNbrDlInitial.64Qam.UEID
    # UE = pd.merge(UE,TBTotNbrDlInitial64QamUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    ##Tot.PdcpSduNbrDl.UEID,
    ##DRB.PdcpSduDelayDl.UEID,
    ##DRB.PdcpSduBitRateDl.UEID,
    ##DRB.PdcpSduVolumeDl_Filter.UEID
    # UE = pd.merge(UE,UELteUEID(DlPdcpStats),on=['Time','IMSI'],how='left').fillna(0)
    #
    ##DRB.IPTimeDL.QOS.UEID
    # UE = pd.merge(UE,DRBIPTimeDLQOSUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #
    ##current_serving_nr
    # rx_df = RxPacketTrace.groupby(['Time','IMSI']).max().reset_index()[['Time','IMSI','time']]
    # rx_df.loc[:,'current_serving_NR'] = rx_df.loc[:,].apply(lambda x: get_serving_cell(x),axis=1)
    # rx_df.drop('time',axis=1,inplace=True)
    # UE = pd.merge(UE,rx_df,on=['Time','IMSI'],how='left')
    ##forward fill the missing values
    # for imsi in range(1,50):
    #    UE.loc[UE['IMSI']==imsi,'current_serving_NR'] = UE.loc[UE['IMSI']==imsi,'current_serving_NR'].fillna(method='ffill')
    #    UE['current_serving_NR'].fillna(8,inplace=True)
    ##RRU.PrbUsedDl.UEID
    # UE = pd.merge(UE,RRUPrbUsedDlUEID(RxPacketTrace),on=['Time','IMSI'],how='left').fillna(0)
    ##DRB.PdcpPduNbrDl.Qos.UEID
    # UE = pd.merge(UE,DRBPdcpPduNbrDlQosUEID(DlRlcStats),on=['Time','IMSI'],how='left').fillna(np.nan)
    ##QosFlow.PdcpPduVolumeDL_Filter.UEID
    # UE = pd.merge(UE,QosFlowPdcpPduVolumeDL_FilterUEID(DlRlcStats),on=['Time','IMSI'],how='left').fillna(np.nan)
    #
    ##***************************CELL level processing*****************************************#
    #
    # CELL = initiate_CELL()
    ##rx_sinr_mcs_cell_df = initial_rx_sinr_mcs_cell(RxPacketTrace)
    ##CARR.PDSCHMCSDist.BinX.BinY.BinZ,L1M.RS-SINR.BinX
    ##CELL = get_SINR_MCS_CELL(CELL,rx_sinr_mcs_cell_df)
    # CELL = pd.merge(CELL,get_SINR_MCS_CELL(RxPacketTrace),on='Time',how='left').fillna(0)
    ##TB.InitialErrNbrDl
    # CELL = pd.merge(CELL,TBInitialErrNbrDl(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    ##TB.TotNbrDl.1
    # CELL = pd.merge(CELL,TBTotNbrDl1(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    ##TB.TotNbrDlInitial
    # CELL = pd.merge(CELL,TBTotNbrDlInitial(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    ##TB.TotNbrDlInitial.16Qam
    # CELL = pd.merge(CELL,TBTotNbrDlInitial16Qam(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    ##TB.TotNbrDlInitial.64Qam
    # CELL = pd.merge(CELL,TBTotNbrDlInitial64Qam(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    ##TB.TotNbrDlInitial.Qpsk
    # CELL = pd.merge(CELL,TBTotNbrDlInitialQpsk(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    ##RRU.PrbUsedDl
    # CELL = pd.merge(CELL,RRUPrbUsedDl(RxPacketTrace),on='Time',how='left').fillna(0)
    ##DRB.MeanActiveUeDl
    # CELL = pd.merge(CELL,DRBMeanActiveUeDl(RxPacketTrace,UeHandoverEndStats),on='Time',how='left')
    ##DL_TotalofAvailablePRBs
    # CELL['DL_TotalofAvailablePRBs'] = 139
    #
    ##******************************Node level processing*************************#
    # NODE = initiate_NODE()
    ##DRB.PdcpSduDelayDl
    # NODE = DRBPdcpSduDelayDl(DlRlcStats)
    ##DRB.UEThpDl
    # NODE = pd.merge(NODE,DRBUEThpDl(DlRlcStats,RxPacketTrace),on='Time')
    ##QosFlow.PdcpPduVolumeDL_Filter
    # NODE = pd.merge(NODE,QosFlowPdcpPduVolumeDL_Filter(DlRlcStats),on='Time',how='left')
    ###*************************** final df**************************************#
    # final_df = pd.merge(pd.merge(UE,CELL,on='Time',how='left'),NODE,on='Time',how='left')
    # final_df['new_current_serving_NR'] = final_df.agg(get_new_serving_cell(UeHandoverStartStats, UeHandoverEndStats)) # new serving cell column
    # final_df.to_csv(os.path.join(data_dir,'final_df.csv'), index=False)
    #
    ##*******Remove temp variables**************#
    # del NODE,CELL,UE,DlPdcpStats,DlRlcStats,UeHandoverStartStats,UeHandoverEndStats,MmWaveSinrTime,RxPacketTrace,RxPacketTrace_DL
    # end_time = datetime.now().strftime("%H:%M:%S")
    # print(f'{cur_proc._identity}: *********{directory} Completed @ {end_time}  ************')


if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #    num_procs = int(sys.argv[1])
    # else:
    #    num_procs = mp.cimport glob
    # import os
    # import pandas as pd
    # import numpy as np
    # from datetime import datetime
    # import multiprocessing as mp
    # import sys
    #
    # import warnings
    # warnings.filterwarnings("ignore")
    #
    #
    # # Define All required functions
    #
    # def getIMSI(row):
    #     condition = ((row['time'] <= UeHandoverStartStats['Time'])
    #                  & (row['cellId'] == UeHandoverStartStats['SourceCellId'])
    #                  & (row['rnti'] == UeHandoverStartStats['RNTI']))
    #     result = UeHandoverStartStats[condition]
    #     if result.shape[0] != 0:
    #         return result.iloc[-1]['IMSI']
    #     condition = ((DlRlcStats['start'] <= row['time'])
    #                  & (DlRlcStats['end'] >= row['time'])
    #                  & (DlRlcStats['CellId'] == row['cellId'])
    #                  & (DlRlcStats['RNTI'] == row['rnti']))
    #     result = DlRlcStats[condition]
    #     if result.shape[0] != 0:
    #         return result.iloc[0]['IMSI']
    #
    # def optimizer_sinr_mcs(row):
    #     final_cell = []
    #     for cell in range(2,9):
    #         condition = ((RxPacketTrace['time']>=row['Time'])
    #                  & (RxPacketTrace['time']<=row['Time']+0.01)
    #                  & (RxPacketTrace['IMSI']==row['IMSI'])
    #                  & (RxPacketTrace['cellId']==cell))
    #         final_cell.append(RxPacketTrace[condition]['SINR(dB)'].values)
    #         final_cell.append(RxPacketTrace[condition]['mcs'].values)
    #     return final_cell[0],final_cell[2],final_cell[4],final_cell[6],final_cell[8],final_cell[10],final_cell[12],            final_cell[1],final_cell[3],final_cell[5],final_cell[7],final_cell[9],final_cell[11],final_cell[13],
    #
    # def bin_sinr(x):
    #     return len(x[(x<-6)])           ,            len(x[(x>=-6) & (x<0)]) ,            len(x[(x>=0) & (x<6)])  ,            len(x[(x>=6) & (x<12)]) ,            len(x[(x>=12) & (x<18)]),            len(x[(x>=18) & (x<24)]),            len(x[(x>=24)])
    #
    # def bin_mcs(x):
    #     return  len(x[(x>=0) & (x<=4)])   ,            len(x[(x>=5) & (x<=9)])   ,            len(x[(x>=10) & (x<=14)]) ,            len(x[(x>=15) & (x<=19)]) ,            len(x[(x>=20) & (x<=24)]) ,            len(x[(x>=25) & (x<=29)])
    #
    #
    # def optimize(row):
    #     val_lst = [0,0,0,0,0,0,0]
    #     if row['cellId'] == 2:
    #         val_lst[0]=row['mcs']
    #     elif row['cellId'] == 3:
    #         val_lst[1]=row['mcs']
    #     elif row['cellId'] == 4:
    #         val_lst[2]=row['mcs']
    #     elif row['cellId'] == 5:
    #         val_lst[3]=row['mcs']
    #     elif row['cellId'] == 6:
    #         val_lst[4]=row['mcs']
    #     elif row['cellId'] == 7:
    #         val_lst[5]=row['mcs']
    #     elif row['cellId'] == 8:
    #         val_lst[6]=row['mcs']
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def optimize_summation(row):
    #     window_size = 0.01
    #     val_lst = [0,0,0,0,0,0,0]
    #     if row['cellId'] == 2:
    #         val_lst[0]=((row['tbSize']*8)/window_size)/1000    #((row['tbSize']*8)/window_size)/1000 #(kbps)
    #     elif row['cellId'] == 3:
    #         val_lst[1]=((row['tbSize']*8)/window_size)/1000
    #     elif row['cellId'] == 4:
    #         val_lst[2]=((row['tbSize']*8)/window_size)/1000
    #     elif row['cellId'] == 5:
    #         val_lst[3]=((row['tbSize']*8)/window_size)/1000
    #     elif row['cellId'] == 6:
    #         val_lst[4]=((row['tbSize']*8)/window_size)/1000
    #     elif row['cellId'] == 7:
    #         val_lst[5]=((row['tbSize']*8)/window_size)/1000
    #     elif row['cellId'] == 8:
    #         val_lst[6]=((row['tbSize']*8)/window_size)/1000
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def max_val(row,rx_df):
    #     return rx_df[(rx_df['Time']==row['Time']) & (rx_df['IMSI']==row['IMSI'])].max().values
    #
    # def drbthpdl(row):
    #     condition= ((row['cellId'] == DlRlcStats['CellId'])
    #             & (row['IMSI'] == DlRlcStats['IMSI'])
    #             & (row['Time'] == DlRlcStats['start']))
    #     result = DlRlcStats[condition]
    #     if not result.empty:
    #         return (result['TxBytes']*8/row['symbol#']).values[0]
    #     return 0
    #
    # def drbthpdl_optimize(row):
    #     val_lst = [0,0,0,0,0,0,0]
    #     if row['cellId'] == 2:
    #         val_lst[0]=row['DRB.IpThpDl.UEID']
    #     elif row['cellId'] == 3:
    #         val_lst[1]=row['DRB.IpThpDl.UEID']
    #     elif row['cellId'] == 4:
    #         val_lst[2]=row['DRB.IpThpDl.UEID']
    #     elif row['cellId'] == 5:
    #         val_lst[3]=row['DRB.IpThpDl.UEID']
    #     elif row['cellId'] == 6:
    #         val_lst[4]=row['DRB.IpThpDl.UEID']
    #     elif row['cellId'] == 7:
    #         val_lst[5]=row['DRB.IpThpDl.UEID']
    #     elif row['cellId'] == 8:
    #         val_lst[6]=row['DRB.IpThpDl.UEID']
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def PdcpPduNbrDl_optimize(row):
    #     val_lst = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    #     if row['CellId'] == 2:
    #         val_lst[0]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     elif row['CellId'] == 3:
    #         val_lst[1]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     elif row['CellId'] == 4:
    #         val_lst[2]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     elif row['CellId'] == 5:
    #         val_lst[3]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     elif row['CellId'] == 6:
    #         val_lst[4]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     elif row['CellId'] == 7:
    #         val_lst[5]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     elif row['CellId'] == 8:
    #         val_lst[6]=row['DRB.PdcpPduNbrDl.Qos.UEID']
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def optimize_IPTimeDL(row):
    #     val_lst = [0,0,0,0,0,0,0]
    #     if row['cellId'] == 2:
    #         val_lst[0]=row['symbol#']
    #     elif row['cellId'] == 3:
    #         val_lst[1]=row['symbol#']
    #     elif row['cellId'] == 4:
    #         val_lst[2]=row['symbol#']
    #     elif row['cellId'] == 5:
    #         val_lst[3]=row['symbol#']
    #     elif row['cellId'] == 6:
    #         val_lst[4]=row['symbol#']
    #     elif row['cellId'] == 7:
    #         val_lst[5]=row['symbol#']
    #     elif row['cellId'] == 8:
    #         val_lst[6]=row['symbol#']
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def optimize_PdcpPduVolumeDL_Filter(row):
    #     val_lst = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    #     if row['CellId'] == 2:
    #         val_lst[0]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     elif row['CellId'] == 3:
    #         val_lst[1]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     elif row['CellId'] == 4:
    #         val_lst[2]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     elif row['CellId'] == 5:
    #         val_lst[3]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     elif row['CellId'] == 6:
    #         val_lst[4]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     elif row['CellId'] == 7:
    #         val_lst[5]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     elif row['CellId'] == 8:
    #         val_lst[6]=row['QosFlow.PdcpPduVolumeDL_Filter.UEID']
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def get_serving_cell(row):
    #     return RxPacketTrace[(row['IMSI'] == RxPacketTrace['IMSI']) & (row['time']==RxPacketTrace['time'])]['cellId'].values[0]
    #
    # def optimize_UE_sp_PRB_schedule(row):
    #     val_lst = [0,0,0,0,0,0,0]
    #     if row['cellId'] == 2:
    #         val_lst[0]=(row['symbol#']/(14*40))*139
    #     elif row['cellId'] == 3:
    #         val_lst[1]=(row['symbol#']/(14*40))*139
    #     elif row['cellId'] == 4:
    #         val_lst[2]=(row['symbol#']/(14*40))*139
    #     elif row['cellId'] == 5:
    #         val_lst[3]=(row['symbol#']/(14*40))*139
    #     elif row['cellId'] == 6:
    #         val_lst[4]=(row['symbol#']/(14*40))*139
    #     elif row['cellId'] == 7:
    #         val_lst[5]=(row['symbol#']/(14*40))*139
    #     elif row['cellId'] == 8:
    #         val_lst[6]=(row['symbol#']/(14*40))*139
    #     return val_lst[0],val_lst[1],val_lst[2],val_lst[3],val_lst[4],val_lst[5],val_lst[6]
    #
    # def get_HO_Cell_Qual_SINR(row):
    #     HO_CellQual_RS_SINR_UEID_Cell2, HO_CellQual_RS_SINR_UEID_Cell3 = None, None
    #     HO_CellQual_RS_SINR_UEID_Cell4, HO_CellQual_RS_SINR_UEID_Cell4 = None, None
    #     HO_CellQual_RS_SINR_UEID_Cell6, HO_CellQual_RS_SINR_UEID_Cell5 = None, None
    #     HO_CellQual_RS_SINR_UEID_Cell8 = None
    #
    #     condition = ((MmWaveSinrTime['Time']<=row['Time'])
    #                  & (MmWaveSinrTime['IMSI'] == row['IMSI']))
    #
    #     max_time = MmWaveSinrTime[condition]['Time'].max()
    #
    #     result = MmWaveSinrTime[(MmWaveSinrTime['Time']==max_time)
    #                & (MmWaveSinrTime['IMSI'] == row['IMSI'])]
    #     for _ , row in result.iterrows():
    #         if row['CellId'] == 2:
    #             HO_CellQual_RS_SINR_UEID_Cell2 = row['SINR']
    #         elif row['CellId'] == 3:
    #             HO_CellQual_RS_SINR_UEID_Cell3 = row['SINR']
    #         elif row['CellId'] == 4:
    #             HO_CellQual_RS_SINR_UEID_Cell4 = row['SINR']
    #         elif row['CellId'] == 5:
    #             HO_CellQual_RS_SINR_UEID_Cell5 = row['SINR']
    #         elif row['CellId'] == 6:
    #             HO_CellQual_RS_SINR_UEID_Cell6 = row['SINR']
    #         elif row['CellId'] == 7:
    #             HO_CellQual_RS_SINR_UEID_Cell7 = row['SINR']
    #         elif row['CellId'] == 8:
    #             HO_CellQual_RS_SINR_UEID_Cell8 = row['SINR']
    #
    #     return [HO_CellQual_RS_SINR_UEID_Cell2,HO_CellQual_RS_SINR_UEID_Cell3,HO_CellQual_RS_SINR_UEID_Cell4,
    #             HO_CellQual_RS_SINR_UEID_Cell5,HO_CellQual_RS_SINR_UEID_Cell6,HO_CellQual_RS_SINR_UEID_Cell7,
    #             HO_CellQual_RS_SINR_UEID_Cell8]
    #
    # def optimizer_sinr_mcs_cell(row):
    #     final_cell = []
    #     for cell in range(2,9):
    #         condition = ((RxPacketTrace['time']>=row['Time'])
    #                  & (RxPacketTrace['time']<=row['Time']+0.01)
    #                  & (RxPacketTrace['cellId']==cell))
    #         final_cell.append(RxPacketTrace[condition]['SINR(dB)'].values)
    #         final_cell.append(RxPacketTrace[condition]['mcs'].values)
    #     return final_cell[0],final_cell[2],final_cell[4],final_cell[6],final_cell[8],final_cell[10],final_cell[12],            final_cell[1],final_cell[3],final_cell[5],final_cell[7],final_cell[9],final_cell[11],final_cell[13],
    #
    # def max_val_cell(row,rx_df):
    #     return rx_df[(rx_df['Time']==row['Time'])].max().values
    #
    # def initial_UE():
    #     ue_df_array = []
    #     time_window =0.01
    #     times = np.arange(0.01, 6.001,time_window)
    #     for imsi in range(1, 50):
    #         ue_df = pd.DataFrame({'Time': times})
    #         ue_df.loc[:, 'IMSI'] = imsi
    #         ue_df_array.append(ue_df)
    #     UE = pd.DataFrame(columns = ue_df.columns)
    #     for dframe in ue_df_array:
    #         UE = UE.append(dframe ,ignore_index=True)
    #     UE['Time'] = UE['Time'].apply(lambda x:round(x,4))
    #     UE['IMSI'] = UE['IMSI'].astype('int32')
    #     return UE
    #
    # def initial_rx_sinr_mcs():
    #     rx_imsi = RxPacketTrace['IMSI'].unique()
    #     rx_time = RxPacketTrace['Time'].unique()
    #     tmp_arry = []
    #     times = rx_time
    #     for imsi in rx_imsi:
    #         temp_df = pd.DataFrame({'Time': times})
    #         temp_df.loc[:, 'IMSI'] = imsi
    #         tmp_arry.append(temp_df)
    #     rx_sinr_mcs_df = pd.DataFrame(columns = temp_df.columns)
    #
    #     for dframe in tmp_arry:
    #         rx_sinr_mcs_df = rx_sinr_mcs_df.append(dframe ,ignore_index=True)
    #     rx_sinr_mcs_df['Time'] = rx_sinr_mcs_df['Time'].apply(lambda x:round(x,4))
    #     rx_sinr_mcs_df['IMSI'] = rx_sinr_mcs_df['IMSI'].astype('int32')
    #     return rx_sinr_mcs_df
    #
    # #def get_SINR_MCS_UEID(UE,rx_sinr_mcs_df):
    # #
    # #    df = pd.DataFrame(columns = ['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #                             'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'])
    # #    rx_sinr_mcs_df = pd.concat([rx_sinr_mcs_df,df])
    # #
    # #    #populate arrays as it's easy for later use
    # #    x = [np.array([],dtype='float64')]*len(rx_sinr_mcs_df)
    # #    rx_sinr_mcs_df[['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #                    'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8']] = \
    # #                    x,x,x,x,x,x,x,x,x,x,x,x,x,x
    # #    rx_sinr_mcs_df.loc[:,['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #                      'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8' ]] = \
    # #                        list(rx_sinr_mcs_df.loc[:,['Time','IMSI']].apply(lambda x:optimizer_sinr_mcs(x),axis=1))
    # #    UE = pd.merge(UE,rx_sinr_mcs_df,on=['Time','IMSI'],how='left')
    # #
    # #    #replace nan with empty array
    # #    for i in UE.columns:
    # #        UE[i].loc[UE[i].isnull()] = [np.array([],dtype='float64')]*len(UE[i].loc[UE[i].isnull()])
    # #
    # #    df = pd.DataFrame(columns = ['L1M.RS-SINR.Bin34.UEID.Cell2','L1M.RS-SINR.Bin46.UEID.Cell2','L1M.RS-SINR.Bin58.UEID.Cell2','L1M.RS-SINR.Bin70.UEID.Cell2',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell2','L1M.RS-SINR.Bin94.UEID.Cell2','L1M.RS-SINR.Bin127.UEID.Cell2',
    # #    'L1M.RS-SINR.Bin34.UEID.Cell3','L1M.RS-SINR.Bin46.UEID.Cell3','L1M.RS-SINR.Bin58.UEID.Cell3','L1M.RS-SINR.Bin70.UEID.Cell3',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell3','L1M.RS-SINR.Bin94.UEID.Cell3','L1M.RS-SINR.Bin127.UEID.Cell3',
    # #    'L1M.RS-SINR.Bin34.UEID.Cell4','L1M.RS-SINR.Bin46.UEID.Cell4','L1M.RS-SINR.Bin58.UEID.Cell4','L1M.RS-SINR.Bin70.UEID.Cell4',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell4','L1M.RS-SINR.Bin94.UEID.Cell4','L1M.RS-SINR.Bin127.UEID.Cell4',
    # #    'L1M.RS-SINR.Bin34.UEID.Cell5','L1M.RS-SINR.Bin46.UEID.Cell5','L1M.RS-SINR.Bin58.UEID.Cell5','L1M.RS-SINR.Bin70.UEID.Cell5',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell5','L1M.RS-SINR.Bin94.UEID.Cell5','L1M.RS-SINR.Bin127.UEID.Cell5',
    # #    'L1M.RS-SINR.Bin34.UEID.Cell6','L1M.RS-SINR.Bin46.UEID.Cell6','L1M.RS-SINR.Bin58.UEID.Cell6','L1M.RS-SINR.Bin70.UEID.Cell6',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell6','L1M.RS-SINR.Bin94.UEID.Cell6','L1M.RS-SINR.Bin127.UEID.Cell6',
    # #    'L1M.RS-SINR.Bin34.UEID.Cell7','L1M.RS-SINR.Bin46.UEID.Cell7','L1M.RS-SINR.Bin58.UEID.Cell7','L1M.RS-SINR.Bin70.UEID.Cell7',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell7','L1M.RS-SINR.Bin94.UEID.Cell7','L1M.RS-SINR.Bin127.UEID.Cell7',
    # #    'L1M.RS-SINR.Bin34.UEID.Cell8','L1M.RS-SINR.Bin46.UEID.Cell8','L1M.RS-SINR.Bin58.UEID.Cell8','L1M.RS-SINR.Bin70.UEID.Cell8',
    # #    'L1M.RS-SINR.Bin82.UEID.Cell8','L1M.RS-SINR.Bin94.UEID.Cell8','L1M.RS-SINR.Bin127.UEID.Cell8',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell2',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell2',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell2',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell2',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell3',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell3',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell3',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell3',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell4',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell4',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell4',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell4',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell5',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell5',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell5',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell5',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell6',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell6',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell6',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell6',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell7',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell7',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell7',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell7',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell8',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell8',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell8',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell8'])
    # #    UE = pd.concat([UE,df])
    # #    #SINR
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell2','L1M.RS-SINR.Bin46.UEID.Cell2', 'L1M.RS-SINR.Bin58.UEID.Cell2',##cell2
    # #          'L1M.RS-SINR.Bin70.UEID.Cell2','L1M.RS-SINR.Bin82.UEID.Cell2','L1M.RS-SINR.Bin94.UEID.Cell2',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell2']] = list(UE.loc[:,'cell_sinr2'].apply(lambda x:bin_sinr(x)))
    # #
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell3','L1M.RS-SINR.Bin46.UEID.Cell3','L1M.RS-SINR.Bin58.UEID.Cell3',#cell3
    # #          'L1M.RS-SINR.Bin70.UEID.Cell3','L1M.RS-SINR.Bin82.UEID.Cell3','L1M.RS-SINR.Bin94.UEID.Cell3',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell3']] = list(UE.loc[:,'cell_sinr3'].apply(lambda x:bin_sinr(x)))
    # #
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell4','L1M.RS-SINR.Bin46.UEID.Cell4','L1M.RS-SINR.Bin58.UEID.Cell4',#cell4
    # #          'L1M.RS-SINR.Bin70.UEID.Cell4','L1M.RS-SINR.Bin82.UEID.Cell4','L1M.RS-SINR.Bin94.UEID.Cell4',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell4']] = list(UE.loc[:,'cell_sinr4'].apply(lambda x:bin_sinr(x)))
    # #
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell5','L1M.RS-SINR.Bin46.UEID.Cell5','L1M.RS-SINR.Bin58.UEID.Cell5',#cell5
    # #          'L1M.RS-SINR.Bin70.UEID.Cell5','L1M.RS-SINR.Bin82.UEID.Cell5','L1M.RS-SINR.Bin94.UEID.Cell5',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell5']] = list(UE.loc[:,'cell_sinr5'].apply(lambda x:bin_sinr(x)))
    # #
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell6','L1M.RS-SINR.Bin46.UEID.Cell6','L1M.RS-SINR.Bin58.UEID.Cell6',#cell6
    # #          'L1M.RS-SINR.Bin70.UEID.Cell6','L1M.RS-SINR.Bin82.UEID.Cell6','L1M.RS-SINR.Bin94.UEID.Cell6',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell6']] = list(UE.loc[:,'cell_sinr6'].apply(lambda x:bin_sinr(x)))
    # #
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell7','L1M.RS-SINR.Bin46.UEID.Cell7','L1M.RS-SINR.Bin58.UEID.Cell7',#cell7
    # #          'L1M.RS-SINR.Bin70.UEID.Cell7','L1M.RS-SINR.Bin82.UEID.Cell7','L1M.RS-SINR.Bin94.UEID.Cell7',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell7']] = list(UE.loc[:,'cell_sinr7'].apply(lambda x:bin_sinr(x)))
    # #
    # #    UE.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell8','L1M.RS-SINR.Bin46.UEID.Cell8','L1M.RS-SINR.Bin58.UEID.Cell8',#cell8
    # #          'L1M.RS-SINR.Bin70.UEID.Cell8','L1M.RS-SINR.Bin82.UEID.Cell8','L1M.RS-SINR.Bin94.UEID.Cell8',
    # #          'L1M.RS-SINR.Bin127.UEID.Cell8']] = list(UE.loc[:,'cell_sinr8'].apply(lambda x:bin_sinr(x)))
    # #    #mcs
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell2',#cell2
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell2',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell2']] = \
    # #            list(UE.loc[:,'cell_mcs2'].apply(lambda x:bin_mcs(x)))
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell3',#cell3
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell3',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell3']] = \
    # #            list(UE.loc[:,'cell_mcs3'].apply(lambda x:bin_mcs(x)))
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell4',#cell4
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell4',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell4']] = \
    # #            list(UE.loc[:,'cell_mcs4'].apply(lambda x:bin_mcs(x)))
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell5',#cell5
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell5',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell5']] = \
    # #            list(UE.loc[:,'cell_mcs5'].apply(lambda x:bin_mcs(x)))
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell6',#cell6
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell6',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell6']] = \
    # #            list(UE.loc[:,'cell_mcs6'].apply(lambda x:bin_mcs(x)))
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell7',#cell7
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell7',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell7']] = \
    # #            list(UE.loc[:,'cell_mcs7'].apply(lambda x:bin_mcs(x)))
    # #    UE.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell8',#cell8
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell8',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell8']] = \
    # #            list(UE.loc[:,'cell_mcs8'].apply(lambda x:bin_mcs(x)))
    # #    UE.drop(['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #         'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'],axis=1,inplace=True)
    # #    return UE
    # def get_SINR_MCS_UEID(rx):
    #     #rx['time_cellid_imsi'] =rx.apply(lambda x: (x['Time'],x['cellId'],x['IMSI']),axis=1)
    #     rx_df = rx[['Time','IMSI','time_cellid_imsi','cellId']]
    #     rx_df.drop_duplicates(inplace=True)
    #     rx_df['sinr'] = rx_df.apply(lambda x:get_new_sinr(x,rx),axis=1)
    #     rx_df['mcs'] = rx_df.apply(lambda x:get_new_mcs(x,rx),axis=1)
    #     rx_df['cell_sinr2'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==2 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr3'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==3 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr4'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==4 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr5'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==5 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr6'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==6 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr7'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==7 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr8'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==8 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs2'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==2 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs3'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==3 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs4'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==4 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs5'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==5 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs6'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==6 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs7'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==7 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs8'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==8 else np.array([],dtype='float64'),axis=1)
    #
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell2','L1M.RS-SINR.Bin46.UEID.Cell2', 'L1M.RS-SINR.Bin58.UEID.Cell2',##cell2
    #           'L1M.RS-SINR.Bin70.UEID.Cell2','L1M.RS-SINR.Bin82.UEID.Cell2','L1M.RS-SINR.Bin94.UEID.Cell2',
    #           'L1M.RS-SINR.Bin127.UEID.Cell2']] = list(rx_df.loc[:,'cell_sinr2'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell3','L1M.RS-SINR.Bin46.UEID.Cell3','L1M.RS-SINR.Bin58.UEID.Cell3',#cell3
    #           'L1M.RS-SINR.Bin70.UEID.Cell3','L1M.RS-SINR.Bin82.UEID.Cell3','L1M.RS-SINR.Bin94.UEID.Cell3',
    #           'L1M.RS-SINR.Bin127.UEID.Cell3']] = list(rx_df.loc[:,'cell_sinr3'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell4','L1M.RS-SINR.Bin46.UEID.Cell4','L1M.RS-SINR.Bin58.UEID.Cell4',#cell4
    #           'L1M.RS-SINR.Bin70.UEID.Cell4','L1M.RS-SINR.Bin82.UEID.Cell4','L1M.RS-SINR.Bin94.UEID.Cell4',
    #           'L1M.RS-SINR.Bin127.UEID.Cell4']] = list(rx_df.loc[:,'cell_sinr4'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell5','L1M.RS-SINR.Bin46.UEID.Cell5','L1M.RS-SINR.Bin58.UEID.Cell5',#cell5
    #           'L1M.RS-SINR.Bin70.UEID.Cell5','L1M.RS-SINR.Bin82.UEID.Cell5','L1M.RS-SINR.Bin94.UEID.Cell5',
    #           'L1M.RS-SINR.Bin127.UEID.Cell5']] = list(rx_df.loc[:,'cell_sinr5'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell6','L1M.RS-SINR.Bin46.UEID.Cell6','L1M.RS-SINR.Bin58.UEID.Cell6',#cell6
    #           'L1M.RS-SINR.Bin70.UEID.Cell6','L1M.RS-SINR.Bin82.UEID.Cell6','L1M.RS-SINR.Bin94.UEID.Cell6',
    #           'L1M.RS-SINR.Bin127.UEID.Cell6']] = list(rx_df.loc[:,'cell_sinr6'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell7','L1M.RS-SINR.Bin46.UEID.Cell7','L1M.RS-SINR.Bin58.UEID.Cell7',#cell7
    #           'L1M.RS-SINR.Bin70.UEID.Cell7','L1M.RS-SINR.Bin82.UEID.Cell7','L1M.RS-SINR.Bin94.UEID.Cell7',
    #           'L1M.RS-SINR.Bin127.UEID.Cell7']] = list(rx_df.loc[:,'cell_sinr7'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.UEID.Cell8','L1M.RS-SINR.Bin46.UEID.Cell8','L1M.RS-SINR.Bin58.UEID.Cell8',#cell8
    #           'L1M.RS-SINR.Bin70.UEID.Cell8','L1M.RS-SINR.Bin82.UEID.Cell8','L1M.RS-SINR.Bin94.UEID.Cell8',
    #           'L1M.RS-SINR.Bin127.UEID.Cell8']] = list(rx_df.loc[:,'cell_sinr8'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell2',#cell2
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell2',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell2']] = \
    #             list(rx_df.loc[:,'cell_mcs2'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell3',#cell3
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell3',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell3']] = \
    #             list(rx_df.loc[:,'cell_mcs3'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell4',#cell4
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell4',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell4']] = \
    #             list(rx_df.loc[:,'cell_mcs4'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell5',#cell5
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell5',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell5']] = \
    #             list(rx_df.loc[:,'cell_mcs5'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell6',#cell6
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell6',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell6']] = \
    #             list(rx_df.loc[:,'cell_mcs6'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell7',#cell7
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell7',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell7']] = \
    #             list(rx_df.loc[:,'cell_mcs7'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.UEID.Cell8',#cell8
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.UEID.Cell8',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.UEID.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.UEID.Cell8']] = \
    #             list(rx_df.loc[:,'cell_mcs8'].apply(lambda x:bin_mcs(x)))
    #
    #     rx_df.drop(['time_cellid_imsi','mcs','sinr','cellId','cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6',
    #                 'cell_sinr7','cell_sinr8','cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6',
    #                 'cell_mcs7','cell_mcs8'],axis=1,inplace=True)
    #     rx_df.loc[:,:] = list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def get_new_sinr(row,df):
    #     return df[df['time_cellid_imsi']==row['time_cellid_imsi']]['SINR(dB)'].values
    #
    # def get_new_mcs(row,df):
    #     return df[df['time_cellid_imsi']==row['time_cellid_imsi']]['mcs'].values
    #
    # def get_new_sinr_cell(row,df):
    #     return df[df['time_cellid']==row['time_cellid']]['SINR(dB)'].values
    #
    # def get_new_mcs_cell(row,df):
    #     return df[df['time_cellid']==row['time_cellid']]['mcs'].values
    #
    # def TBTotNbrDl1UEID(rx_DL):
    #     rx_df = rx_DL.groupby(['cellId','IMSI','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','TB.TotNbrDl.1.UEID.Cell2','TB.TotNbrDl.1.UEID.Cell3','TB.TotNbrDl.1.UEID.Cell4',
    #              'TB.TotNbrDl.1.UEID.Cell5','TB.TotNbrDl.1.UEID.Cell6','TB.TotNbrDl.1.UEID.Cell7','TB.TotNbrDl.1.UEID.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDl.1.UEID.Cell2','TB.TotNbrDl.1.UEID.Cell3','TB.TotNbrDl.1.UEID.Cell4',
    #              'TB.TotNbrDl.1.UEID.Cell5','TB.TotNbrDl.1.UEID.Cell6','TB.TotNbrDl.1.UEID.Cell7','TB.TotNbrDl.1.UEID.Cell8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','TB.TotNbrDl.1.UEID.Cell2','TB.TotNbrDl.1.UEID.Cell3','TB.TotNbrDl.1.UEID.Cell4',
    #              'TB.TotNbrDl.1.UEID.Cell5','TB.TotNbrDl.1.UEID.Cell6','TB.TotNbrDl.1.UEID.Cell7','TB.TotNbrDl.1.UEID.Cell8']]
    #
    #     rx_df.loc[:,['Time','IMSI','TB.TotNbrDl.1.UEID.Cell2','TB.TotNbrDl.1.UEID.Cell3','TB.TotNbrDl.1.UEID.Cell4',
    #              'TB.TotNbrDl.1.UEID.Cell5','TB.TotNbrDl.1.UEID.Cell6','TB.TotNbrDl.1.UEID.Cell7','TB.TotNbrDl.1.UEID.Cell8']] = \
    #                 list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitialUEID(rx_DL):
    #     rx_df = rx_DL[rx_DL['rv']==0].groupby(['cellId','IMSI','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','TB.TotNbrDlInitial.UEID.Cell2','TB.TotNbrDlInitial.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.UEID.Cell4','TB.TotNbrDlInitial.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.UEID.Cell6','TB.TotNbrDlInitial.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.UEID.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDlInitial.UEID.Cell2','TB.TotNbrDlInitial.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.UEID.Cell4','TB.TotNbrDlInitial.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.UEID.Cell6','TB.TotNbrDlInitial.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.UEID.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #     rx_df = rx_df[['Time','IMSI','TB.TotNbrDlInitial.UEID.Cell2','TB.TotNbrDlInitial.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.UEID.Cell4','TB.TotNbrDlInitial.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.UEID.Cell6','TB.TotNbrDlInitial.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.UEID.Cell8']]
    #     rx_df.loc[:,['Time','IMSI','TB.TotNbrDlInitial.UEID.Cell2','TB.TotNbrDlInitial.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.UEID.Cell4','TB.TotNbrDlInitial.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.UEID.Cell6','TB.TotNbrDlInitial.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.UEID.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def DRBPdcpBitrateQOSUEID(rx_DL):
    #     rx_df = rx_DL[rx_DL['rv']==0].groupby(['cellId','IMSI','Time']).sum().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','DRB.PdcpBitrate.QOS.UEID.Node2','DRB.PdcpBitrate.QOS.UEID.Node3',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node4','DRB.PdcpBitrate.QOS.UEID.Node5',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node6','DRB.PdcpBitrate.QOS.UEID.Node7',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node8'])
    #     rx_df.loc[:,['DRB.PdcpBitrate.QOS.UEID.Node2','DRB.PdcpBitrate.QOS.UEID.Node3',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node4','DRB.PdcpBitrate.QOS.UEID.Node5',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node6','DRB.PdcpBitrate.QOS.UEID.Node7',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize_summation(x),axis=1))
    #     rx_df = rx_df[['Time','IMSI','DRB.PdcpBitrate.QOS.UEID.Node2','DRB.PdcpBitrate.QOS.UEID.Node3',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node4','DRB.PdcpBitrate.QOS.UEID.Node5',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node6','DRB.PdcpBitrate.QOS.UEID.Node7',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node8']]
    #     rx_df.loc[:,['Time','IMSI','DRB.PdcpBitrate.QOS.UEID.Node2','DRB.PdcpBitrate.QOS.UEID.Node3',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node4','DRB.PdcpBitrate.QOS.UEID.Node5',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node6','DRB.PdcpBitrate.QOS.UEID.Node7',
    #                                             'DRB.PdcpBitrate.QOS.UEID.Node8']] = \
    #             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBInitialErrNbrDlUEID(rx_DL):
    #     rx_df = rx_DL[rx_DL['rv']!=0].groupby(['cellId','IMSI','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','TB.InitialErrNbrDl.UEID.Cell2','TB.InitialErrNbrDl.UEID.Cell3',
    #                                             'TB.InitialErrNbrDl.UEID.Cell4','TB.InitialErrNbrDl.UEID.Cell5',
    #                                             'TB.InitialErrNbrDl.UEID.Cell6','TB.InitialErrNbrDl.UEID.Cell7',
    #                                             'TB.InitialErrNbrDl.UEID.Cell8'])
    #
    #     rx_df.loc[:,['TB.InitialErrNbrDl.UEID.Cell2','TB.InitialErrNbrDl.UEID.Cell3',
    #                                             'TB.InitialErrNbrDl.UEID.Cell4','TB.InitialErrNbrDl.UEID.Cell5',
    #                                             'TB.InitialErrNbrDl.UEID.Cell6','TB.InitialErrNbrDl.UEID.Cell7',
    #                                             'TB.InitialErrNbrDl.UEID.Cell8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #     rx_df = rx_df[['Time','IMSI','TB.InitialErrNbrDl.UEID.Cell2','TB.InitialErrNbrDl.UEID.Cell3',
    #                                             'TB.InitialErrNbrDl.UEID.Cell4','TB.InitialErrNbrDl.UEID.Cell5',
    #                                             'TB.InitialErrNbrDl.UEID.Cell6','TB.InitialErrNbrDl.UEID.Cell7',
    #                                             'TB.InitialErrNbrDl.UEID.Cell8']]
    #     rx_df.loc[:,['Time','IMSI','TB.InitialErrNbrDl.UEID.Cell2','TB.InitialErrNbrDl.UEID.Cell3',
    #                                             'TB.InitialErrNbrDl.UEID.Cell4','TB.InitialErrNbrDl.UEID.Cell5',
    #                                             'TB.InitialErrNbrDl.UEID.Cell6','TB.InitialErrNbrDl.UEID.Cell7',
    #                                             'TB.InitialErrNbrDl.UEID.Cell8']] = \
    #             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitialQpskUEID(rx_DL):
    #     rx_df = rx_DL[rx_DL['mcs'].isin([0,1,2,3,4,5,6,7,8,9])].groupby(['cellId','IMSI','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','TB.TotNbrDlInitial.Qpsk.UEID.Cell2','TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell4','TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell6','TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDlInitial.Qpsk.UEID.Cell2','TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell4','TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell6','TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','TB.TotNbrDlInitial.Qpsk.UEID.Cell2','TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell4','TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell6','TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell8']]
    #
    #     rx_df.loc[:,['Time','IMSI','TB.TotNbrDlInitial.Qpsk.UEID.Cell2','TB.TotNbrDlInitial.Qpsk.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell4','TB.TotNbrDlInitial.Qpsk.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell6','TB.TotNbrDlInitial.Qpsk.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.UEID.Cell8']] = \
    #             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitial16QamUEID(rx_DL):
    #     rx_df = rx_DL[rx_DL['mcs'].isin([10,11,12,13,14,15,16])].groupby(['cellId','IMSI','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','TB.TotNbrDlInitial.16Qam.UEID.Cell2','TB.TotNbrDlInitial.16Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell4','TB.TotNbrDlInitial.16Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell6','TB.TotNbrDlInitial.16Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDlInitial.16Qam.UEID.Cell2','TB.TotNbrDlInitial.16Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell4','TB.TotNbrDlInitial.16Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell6','TB.TotNbrDlInitial.16Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','TB.TotNbrDlInitial.16Qam.UEID.Cell2','TB.TotNbrDlInitial.16Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell4','TB.TotNbrDlInitial.16Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell6','TB.TotNbrDlInitial.16Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell8']]
    #
    #     rx_df.loc[:,['Time','IMSI','TB.TotNbrDlInitial.16Qam.UEID.Cell2','TB.TotNbrDlInitial.16Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell4','TB.TotNbrDlInitial.16Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell6','TB.TotNbrDlInitial.16Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.UEID.Cell8']] = \
    #             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitial64QamUEID(rx_DL):
    #     rx_df = rx_DL[rx_DL['mcs'].isin([17,18,19,20,21,22,23,24,25,26,27,28])].groupby(['cellId','IMSI','Time']).count().reset_index()
    #
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','TB.TotNbrDlInitial.64Qam.UEID.Cell2','TB.TotNbrDlInitial.64Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell4','TB.TotNbrDlInitial.64Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell6','TB.TotNbrDlInitial.64Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell8'])
    #     rx_df.loc[:,['TB.TotNbrDlInitial.64Qam.UEID.Cell2','TB.TotNbrDlInitial.64Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell4','TB.TotNbrDlInitial.64Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell6','TB.TotNbrDlInitial.64Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','TB.TotNbrDlInitial.64Qam.UEID.Cell2','TB.TotNbrDlInitial.64Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell4','TB.TotNbrDlInitial.64Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell6','TB.TotNbrDlInitial.64Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell8']]
    #
    #     rx_df.loc[:,['Time','IMSI','TB.TotNbrDlInitial.64Qam.UEID.Cell2','TB.TotNbrDlInitial.64Qam.UEID.Cell3',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell4','TB.TotNbrDlInitial.64Qam.UEID.Cell5',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell6','TB.TotNbrDlInitial.64Qam.UEID.Cell7',
    #                                             'TB.TotNbrDlInitial.64Qam.UEID.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def UELteUEID(dlpdcp):
    #     dlpdcp['DRB.PdcpSduDelayDl.UEID'] = dlpdcp['delay']*10000
    #     dlpdcp['DRB.PdcpSduBitRateDl.UEID'] = ((dlpdcp['TxBytes']*8)/(0.01))/1000
    #     dlpdcp['DRB.PdcpSduVolumeDl_Filter.UEID'] = (dlpdcp['TxBytes']*8)/(10**3)
    #     rx_df = dlpdcp[['start','IMSI','nTxPDUs','DRB.PdcpSduDelayDl.UEID',
    #                     'DRB.PdcpSduBitRateDl.UEID','DRB.PdcpSduVolumeDl_Filter.UEID']]
    #     rx_df.rename(columns={'start':'Time','nTxPDUs':'Tot.PdcpSduNbrDl.UEID'},inplace=True)
    #     dlpdcp.drop(['DRB.PdcpSduDelayDl.UEID','DRB.PdcpSduBitRateDl.UEID','DRB.PdcpSduVolumeDl_Filter.UEID'],axis=1,inplace=True)
    #     return rx_df
    #
    # def DRBPdcpPduNbrDlQosUEID(DlRlcStats):
    #     rx_df = DlRlcStats[['start','IMSI','CellId','nTxPDUs']]
    #     rx_df.rename(columns = {'start':'Time','nTxPDUs':'DRB.PdcpPduNbrDl.Qos.UEID'},inplace=True)
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','DRB.PdcpPduNbrDl.Qos.UEID.Node2','DRB.PdcpPduNbrDl.Qos.UEID.Node3',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node4','DRB.PdcpPduNbrDl.Qos.UEID.Node5',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node6','DRB.PdcpPduNbrDl.Qos.UEID.Node7',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node8'])
    #     rx_df.loc[:,['DRB.PdcpPduNbrDl.Qos.UEID.Node2','DRB.PdcpPduNbrDl.Qos.UEID.Node3',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node4','DRB.PdcpPduNbrDl.Qos.UEID.Node5',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node6','DRB.PdcpPduNbrDl.Qos.UEID.Node7',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:PdcpPduNbrDl_optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','DRB.PdcpPduNbrDl.Qos.UEID.Node2','DRB.PdcpPduNbrDl.Qos.UEID.Node3',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node4','DRB.PdcpPduNbrDl.Qos.UEID.Node5',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node6','DRB.PdcpPduNbrDl.Qos.UEID.Node7',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node8']]
    #
    #     rx_df.loc[:,['Time','IMSI','DRB.PdcpPduNbrDl.Qos.UEID.Node2','DRB.PdcpPduNbrDl.Qos.UEID.Node3',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node4','DRB.PdcpPduNbrDl.Qos.UEID.Node5',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node6','DRB.PdcpPduNbrDl.Qos.UEID.Node7',
    #                                             'DRB.PdcpPduNbrDl.Qos.UEID.Node8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def DRBIPTimeDLQOSUEID(rx_DL):
    #     rx_df = rx_DL.groupby(['cellId','IMSI','Time']).sum().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','DRB.IPTimeDL.QOS.UEID.Cell2','DRB.IPTimeDL.QOS.UEID.Cell3',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell4','DRB.IPTimeDL.QOS.UEID.Cell5',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell6','DRB.IPTimeDL.QOS.UEID.Cell7',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell8'])
    #     rx_df.loc[:,['DRB.IPTimeDL.QOS.UEID.Cell2','DRB.IPTimeDL.QOS.UEID.Cell3',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell4','DRB.IPTimeDL.QOS.UEID.Cell5',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell6','DRB.IPTimeDL.QOS.UEID.Cell7',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize_IPTimeDL(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','DRB.IPTimeDL.QOS.UEID.Cell2','DRB.IPTimeDL.QOS.UEID.Cell3',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell4','DRB.IPTimeDL.QOS.UEID.Cell5',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell6','DRB.IPTimeDL.QOS.UEID.Cell7',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell8']]
    #
    #     rx_df.loc[:,['Time','IMSI','DRB.IPTimeDL.QOS.UEID.Cell2','DRB.IPTimeDL.QOS.UEID.Cell3',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell4','DRB.IPTimeDL.QOS.UEID.Cell5',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell6','DRB.IPTimeDL.QOS.UEID.Cell7',
    #                                             'DRB.IPTimeDL.QOS.UEID.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def QosFlowPdcpPduVolumeDL_FilterUEID(DlRlcStats):
    #     rx_df = DlRlcStats[['start','IMSI','CellId','TxBytes']]
    #     rx_df['TxBytes'] = (rx_df['TxBytes']*8)/10**3
    #     rx_df.rename(columns={'start':'Time','TxBytes':'QosFlow.PdcpPduVolumeDL_Filter.UEID'},inplace=True)
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8'])
    #
    #     rx_df.loc[:,['QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8']] = \
    #                              list(rx_df.loc[:,].apply(lambda x:optimize_PdcpPduVolumeDL_Filter(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8']]
    #
    #     rx_df.loc[:,['Time','IMSI','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node2','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node4','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node6','QosFlow.PdcpPduVolumeDL_Filter.UEID.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.UEID.Node8']] = \
    #                             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def RRUPrbUsedDlUEID(RxPacketTrace):
    #     #Periodicity = 10*4
    #     DR = 10*4*14
    #     rx_df = RxPacketTrace.groupby(['cellId','IMSI','Time']).sum().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','IMSI','RRU.PrbUsedDl.UEID.Cell2','RRU.PrbUsedDl.UEID.Cell3',
    #                                             'RRU.PrbUsedDl.UEID.Cell4','RRU.PrbUsedDl.UEID.Cell5',
    #                                             'RRU.PrbUsedDl.UEID.Cell6','RRU.PrbUsedDl.UEID.Cell7',
    #                                             'RRU.PrbUsedDl.UEID.Cell8'])
    #
    #     rx_df.loc[:,['RRU.PrbUsedDl.UEID.Cell2','RRU.PrbUsedDl.UEID.Cell3',
    #                                             'RRU.PrbUsedDl.UEID.Cell4','RRU.PrbUsedDl.UEID.Cell5',
    #                                             'RRU.PrbUsedDl.UEID.Cell6','RRU.PrbUsedDl.UEID.Cell7',
    #                                             'RRU.PrbUsedDl.UEID.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize_UE_sp_PRB_schedule(x),axis=1))
    #
    #     rx_df = rx_df[['Time','IMSI','RRU.PrbUsedDl.UEID.Cell2','RRU.PrbUsedDl.UEID.Cell3',
    #                                             'RRU.PrbUsedDl.UEID.Cell4','RRU.PrbUsedDl.UEID.Cell5',
    #                                             'RRU.PrbUsedDl.UEID.Cell6','RRU.PrbUsedDl.UEID.Cell7',
    #                                             'RRU.PrbUsedDl.UEID.Cell8']]
    #
    #     rx_df.loc[:,['Time','IMSI','RRU.PrbUsedDl.UEID.Cell2','RRU.PrbUsedDl.UEID.Cell3',
    #                                             'RRU.PrbUsedDl.UEID.Cell4','RRU.PrbUsedDl.UEID.Cell5',
    #                                             'RRU.PrbUsedDl.UEID.Cell6','RRU.PrbUsedDl.UEID.Cell7',
    #                                             'RRU.PrbUsedDl.UEID.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # # Cell level functions
    #
    # def initiate_CELL():
    #     time_window =0.01
    #     times = np.arange(0.01, 6.001,time_window)
    #     CELL = pd.DataFrame({'Time': times})
    #     CELL['Time'] = CELL['Time'].apply(lambda x:round(x,4))
    #     return CELL
    #
    # def initial_rx_sinr_mcs_cell(RxPacketTrace):
    #     rx_time = RxPacketTrace['Time'].unique()
    #     rx_sinr_mcs_df = pd.DataFrame({'Time': rx_time})
    #     rx_sinr_mcs_df['Time'] = rx_sinr_mcs_df['Time'].apply(lambda x:round(x,4))
    #     return rx_sinr_mcs_df
    #
    # #def get_SINR_MCS_CELL(CELL,rx_sinr_mcs_df):
    # #
    # #    df = pd.DataFrame(columns = ['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #                             'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'])
    # #    rx_sinr_mcs_df = pd.concat([rx_sinr_mcs_df,df])
    # #
    # #    #populate arrays as it's easy for later use
    # #    x = [np.array([],dtype='float64')]*len(rx_sinr_mcs_df)
    # #    rx_sinr_mcs_df[['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #                    'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8']] = \
    # #                    x,x,x,x,x,x,x,x,x,x,x,x,x,x
    # #    rx_sinr_mcs_df.loc[:,['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #                      'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8' ]] = \
    # #                list(rx_sinr_mcs_df.loc[:,['Time']].apply(lambda x:optimizer_sinr_mcs_cell(x),axis=1))
    # #    CELL = pd.merge(CELL,rx_sinr_mcs_df,on=['Time'],how='left')
    # #
    # #    #replace nan with empty array
    # #    for i in CELL.columns:
    # #        CELL[i].loc[CELL[i].isnull()] = [np.array([],dtype='float64')]*len(CELL[i].loc[CELL[i].isnull()])
    # #
    # #    df = pd.DataFrame(columns = ['L1M.RS-SINR.Bin34.Cell2','L1M.RS-SINR.Bin46.Cell2','L1M.RS-SINR.Bin58.Cell2','L1M.RS-SINR.Bin70.Cell2',
    # #    'L1M.RS-SINR.Bin82.Cell2','L1M.RS-SINR.Bin94.Cell2','L1M.RS-SINR.Bin127.Cell2',
    # #    'L1M.RS-SINR.Bin34.Cell3','L1M.RS-SINR.Bin46.Cell3','L1M.RS-SINR.Bin58.Cell3','L1M.RS-SINR.Bin70.Cell3',
    # #    'L1M.RS-SINR.Bin82.Cell3','L1M.RS-SINR.Bin94.Cell3','L1M.RS-SINR.Bin127.Cell3',
    # #    'L1M.RS-SINR.Bin34.Cell4','L1M.RS-SINR.Bin46.Cell4','L1M.RS-SINR.Bin58.Cell4','L1M.RS-SINR.Bin70.Cell4',
    # #    'L1M.RS-SINR.Bin82.Cell4','L1M.RS-SINR.Bin94.Cell4','L1M.RS-SINR.Bin127.Cell4',
    # #    'L1M.RS-SINR.Bin34.Cell5','L1M.RS-SINR.Bin46.Cell5','L1M.RS-SINR.Bin58.Cell5','L1M.RS-SINR.Bin70.Cell5',
    # #    'L1M.RS-SINR.Bin82.Cell5','L1M.RS-SINR.Bin94.Cell5','L1M.RS-SINR.Bin127.Cell5',
    # #    'L1M.RS-SINR.Bin34.Cell6','L1M.RS-SINR.Bin46.Cell6','L1M.RS-SINR.Bin58.Cell6','L1M.RS-SINR.Bin70.Cell6',
    # #    'L1M.RS-SINR.Bin82.Cell6','L1M.RS-SINR.Bin94.Cell6','L1M.RS-SINR.Bin127.Cell6',
    # #    'L1M.RS-SINR.Bin34.Cell7','L1M.RS-SINR.Bin46.Cell7','L1M.RS-SINR.Bin58.Cell7','L1M.RS-SINR.Bin70.Cell7',
    # #    'L1M.RS-SINR.Bin82.Cell7','L1M.RS-SINR.Bin94.Cell7','L1M.RS-SINR.Bin127.Cell7',
    # #    'L1M.RS-SINR.Bin34.Cell8','L1M.RS-SINR.Bin46.Cell8','L1M.RS-SINR.Bin58.Cell8','L1M.RS-SINR.Bin70.Cell8',
    # #    'L1M.RS-SINR.Bin82.Cell8','L1M.RS-SINR.Bin94.Cell8','L1M.RS-SINR.Bin127.Cell8',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell2',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell2',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell2',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell2',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell3',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell3',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell3',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell3',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell4',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell4',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell4',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell4',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell5',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell5',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell5',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell5',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell6',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell6',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell6',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell6',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell7',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell7',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell7',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell7',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell8',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell8',
    # #    'CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell8',
    # #                           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell8'])
    # #    CELL = pd.concat([CELL,df])
    # #    #SINR
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell2','L1M.RS-SINR.Bin46.Cell2', 'L1M.RS-SINR.Bin58.Cell2',##cell2
    # #          'L1M.RS-SINR.Bin70.Cell2','L1M.RS-SINR.Bin82.Cell2','L1M.RS-SINR.Bin94.Cell2',
    # #          'L1M.RS-SINR.Bin127.Cell2']] = list(CELL.loc[:,'cell_sinr2'].apply(lambda x:bin_sinr(x)))
    # #
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell3','L1M.RS-SINR.Bin46.Cell3','L1M.RS-SINR.Bin58.Cell3',#cell3
    # #          'L1M.RS-SINR.Bin70.Cell3','L1M.RS-SINR.Bin82.Cell3','L1M.RS-SINR.Bin94.Cell3',
    # #          'L1M.RS-SINR.Bin127.Cell3']] = list(CELL.loc[:,'cell_sinr3'].apply(lambda x:bin_sinr(x)))
    # #
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell4','L1M.RS-SINR.Bin46.Cell4','L1M.RS-SINR.Bin58.Cell4',#cell4
    # #          'L1M.RS-SINR.Bin70.Cell4','L1M.RS-SINR.Bin82.Cell4','L1M.RS-SINR.Bin94.Cell4',
    # #          'L1M.RS-SINR.Bin127.Cell4']] = list(CELL.loc[:,'cell_sinr4'].apply(lambda x:bin_sinr(x)))
    # #
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell5','L1M.RS-SINR.Bin46.Cell5','L1M.RS-SINR.Bin58.Cell5',#cell5
    # #          'L1M.RS-SINR.Bin70.Cell5','L1M.RS-SINR.Bin82.Cell5','L1M.RS-SINR.Bin94.Cell5',
    # #          'L1M.RS-SINR.Bin127.Cell5']] = list(CELL.loc[:,'cell_sinr5'].apply(lambda x:bin_sinr(x)))
    # #
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell6','L1M.RS-SINR.Bin46.Cell6','L1M.RS-SINR.Bin58.Cell6',#cell6
    # #          'L1M.RS-SINR.Bin70.Cell6','L1M.RS-SINR.Bin82.Cell6','L1M.RS-SINR.Bin94.Cell6',
    # #          'L1M.RS-SINR.Bin127.Cell6']] = list(CELL.loc[:,'cell_sinr6'].apply(lambda x:bin_sinr(x)))
    # #
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell7','L1M.RS-SINR.Bin46.Cell7','L1M.RS-SINR.Bin58.Cell7',#cell7
    # #          'L1M.RS-SINR.Bin70.Cell7','L1M.RS-SINR.Bin82.Cell7','L1M.RS-SINR.Bin94.Cell7',
    # #          'L1M.RS-SINR.Bin127.Cell7']] = list(CELL.loc[:,'cell_sinr7'].apply(lambda x:bin_sinr(x)))
    # #
    # #    CELL.loc[:,['L1M.RS-SINR.Bin34.Cell8','L1M.RS-SINR.Bin46.Cell8','L1M.RS-SINR.Bin58.Cell8',#cell8
    # #          'L1M.RS-SINR.Bin70.Cell8','L1M.RS-SINR.Bin82.Cell8','L1M.RS-SINR.Bin94.Cell8',
    # #          'L1M.RS-SINR.Bin127.Cell8']] = list(CELL.loc[:,'cell_sinr8'].apply(lambda x:bin_sinr(x)))
    # #    #mcs
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell2',#cell2
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell2',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell2']] = \
    # #            list(CELL.loc[:,'cell_mcs2'].apply(lambda x:bin_mcs(x)))
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell3',#cell3
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell3',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell3']] = \
    # #            list(CELL.loc[:,'cell_mcs3'].apply(lambda x:bin_mcs(x)))
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell4',#cell4
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell4',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell4']] = \
    # #            list(CELL.loc[:,'cell_mcs4'].apply(lambda x:bin_mcs(x)))
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell5',#cell5
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell5',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell5']] = \
    # #            list(CELL.loc[:,'cell_mcs5'].apply(lambda x:bin_mcs(x)))
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell6',#cell6
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell6',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell6']] = \
    # #            list(CELL.loc[:,'cell_mcs6'].apply(lambda x:bin_mcs(x)))
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell7',#cell7
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell7',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell7']] = \
    # #            list(CELL.loc[:,'cell_mcs7'].apply(lambda x:bin_mcs(x)))
    # #    CELL.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell8',#cell8
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell8',
    # #          'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell8']] = \
    # #            list(CELL.loc[:,'cell_mcs8'].apply(lambda x:bin_mcs(x)))
    # #    CELL.drop(['cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6','cell_sinr7','cell_sinr8',
    # #         'cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6','cell_mcs7','cell_mcs8'],axis=1,inplace=True)
    # #    return CELL
    #
    # def get_SINR_MCS_CELL(rx):
    #     rx_df = rx[['Time','time_cellid','cellId']]
    #     rx_df.drop_duplicates(inplace=True)
    #     rx_df['sinr'] = rx_df.apply(lambda x:get_new_sinr_cell(x,rx),axis=1)
    #     rx_df['mcs'] = rx_df.apply(lambda x:get_new_mcs_cell(x,rx),axis=1)
    #     rx_df['cell_sinr2'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==2 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr3'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==3 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr4'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==4 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr5'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==5 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr6'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==6 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr7'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==7 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_sinr8'] = rx_df.apply(lambda x: x['sinr'] if x['cellId']==8 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs2'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==2 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs3'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==3 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs4'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==4 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs5'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==5 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs6'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==6 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs7'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==7 else np.array([],dtype='float64'),axis=1)
    #     rx_df['cell_mcs8'] = rx_df.apply(lambda x: x['mcs'] if x['cellId']==8 else np.array([],dtype='float64'),axis=1)
    #
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell2','L1M.RS-SINR.Bin46.Cell2', 'L1M.RS-SINR.Bin58.Cell2',##cell2
    #           'L1M.RS-SINR.Bin70.Cell2','L1M.RS-SINR.Bin82.Cell2','L1M.RS-SINR.Bin94.Cell2',
    #           'L1M.RS-SINR.Bin127.Cell2']] = list(rx_df.loc[:,'cell_sinr2'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell3','L1M.RS-SINR.Bin46.Cell3','L1M.RS-SINR.Bin58.Cell3',#cell3
    #           'L1M.RS-SINR.Bin70.Cell3','L1M.RS-SINR.Bin82.Cell3','L1M.RS-SINR.Bin94.Cell3',
    #           'L1M.RS-SINR.Bin127.Cell3']] = list(rx_df.loc[:,'cell_sinr3'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell4','L1M.RS-SINR.Bin46.Cell4','L1M.RS-SINR.Bin58.Cell4',#cell4
    #           'L1M.RS-SINR.Bin70.Cell4','L1M.RS-SINR.Bin82.Cell4','L1M.RS-SINR.Bin94.Cell4',
    #           'L1M.RS-SINR.Bin127.Cell4']] = list(rx_df.loc[:,'cell_sinr4'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell5','L1M.RS-SINR.Bin46.Cell5','L1M.RS-SINR.Bin58.Cell5',#cell5
    #           'L1M.RS-SINR.Bin70.Cell5','L1M.RS-SINR.Bin82.Cell5','L1M.RS-SINR.Bin94.Cell5',
    #           'L1M.RS-SINR.Bin127.Cell5']] = list(rx_df.loc[:,'cell_sinr5'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell6','L1M.RS-SINR.Bin46.Cell6','L1M.RS-SINR.Bin58.Cell6',#cell6
    #           'L1M.RS-SINR.Bin70.Cell6','L1M.RS-SINR.Bin82.Cell6','L1M.RS-SINR.Bin94.Cell6',
    #           'L1M.RS-SINR.Bin127.Cell6']] = list(rx_df.loc[:,'cell_sinr6'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell7','L1M.RS-SINR.Bin46.Cell7','L1M.RS-SINR.Bin58.Cell7',#cell7
    #           'L1M.RS-SINR.Bin70.Cell7','L1M.RS-SINR.Bin82.Cell7','L1M.RS-SINR.Bin94.Cell7',
    #           'L1M.RS-SINR.Bin127.Cell7']] = list(rx_df.loc[:,'cell_sinr7'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['L1M.RS-SINR.Bin34.Cell8','L1M.RS-SINR.Bin46.Cell8','L1M.RS-SINR.Bin58.Cell8',#cell8
    #           'L1M.RS-SINR.Bin70.Cell8','L1M.RS-SINR.Bin82.Cell8','L1M.RS-SINR.Bin94.Cell8',
    #           'L1M.RS-SINR.Bin127.Cell8']] = list(rx_df.loc[:,'cell_sinr8'].apply(lambda x:bin_sinr(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell2',#cell2
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell2',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell2','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell2']] = \
    #             list(rx_df.loc[:,'cell_mcs2'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell3',#cell3
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell3',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell3','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell3']] = \
    #             list(rx_df.loc[:,'cell_mcs3'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell4',#cell4
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell4',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell4','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell4']] = \
    #             list(rx_df.loc[:,'cell_mcs4'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell5',#cell5
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell5',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell5','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell5']] = \
    #             list(rx_df.loc[:,'cell_mcs5'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell6',#cell6
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell6',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell6','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell6']] = \
    #             list(rx_df.loc[:,'cell_mcs6'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell7',#cell7
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell7',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell7','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell7']] = \
    #             list(rx_df.loc[:,'cell_mcs7'].apply(lambda x:bin_mcs(x)))
    #     rx_df.loc[:,['CARR.PDSCHMCSDist.Bin1.Bin1.Bin4.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin10.Cell8',#cell8
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin15.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin20.Cell8',
    #           'CARR.PDSCHMCSDist.Bin1.Bin1.Bin25.Cell8','CARR.PDSCHMCSDist.Bin1.Bin1.Bin30.Cell8']] = \
    #             list(rx_df.loc[:,'cell_mcs8'].apply(lambda x:bin_mcs(x)))
    #
    #     rx_df.drop(['time_cellid','mcs','sinr','cellId','cell_sinr2','cell_sinr3','cell_sinr4','cell_sinr5','cell_sinr6',
    #                 'cell_sinr7','cell_sinr8','cell_mcs2','cell_mcs3','cell_mcs4','cell_mcs5','cell_mcs6',
    #                 'cell_mcs7','cell_mcs8'],axis=1,inplace=True)
    #     rx_df.loc[:,:] = list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBInitialErrNbrDl(rx_DL):
    #     rx_df = rx_DL[rx_DL['rv']!=0].groupby(['cellId','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(['Time','TB.InitialErrNbrDl.Cell2','TB.InitialErrNbrDl.Cell3',
    #                                             'TB.InitialErrNbrDl.Cell4','TB.InitialErrNbrDl.Cell5',
    #                                             'TB.InitialErrNbrDl.Cell6','TB.InitialErrNbrDl.Cell7',
    #                                             'TB.InitialErrNbrDl.Cell8'])
    #     rx_df.loc[:,['TB.InitialErrNbrDl.Cell2','TB.InitialErrNbrDl.Cell3',
    #                                             'TB.InitialErrNbrDl.Cell4','TB.InitialErrNbrDl.Cell5',
    #                                             'TB.InitialErrNbrDl.Cell6','TB.InitialErrNbrDl.Cell7',
    #                                             'TB.InitialErrNbrDl.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #     rx_df = rx_df[['Time','TB.InitialErrNbrDl.Cell2','TB.InitialErrNbrDl.Cell3',
    #                                             'TB.InitialErrNbrDl.Cell4','TB.InitialErrNbrDl.Cell5',
    #                                             'TB.InitialErrNbrDl.Cell6','TB.InitialErrNbrDl.Cell7',
    #                                             'TB.InitialErrNbrDl.Cell8']]
    #     rx_df.loc[:,['Time','TB.InitialErrNbrDl.Cell2','TB.InitialErrNbrDl.Cell3',
    #                                             'TB.InitialErrNbrDl.Cell4','TB.InitialErrNbrDl.Cell5',
    #                                             'TB.InitialErrNbrDl.Cell6','TB.InitialErrNbrDl.Cell7',
    #                                             'TB.InitialErrNbrDl.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDl1(rx_DL):
    #     rx_df = rx_DL.groupby(['cellId','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','TB.TotNbrDl.1.Cell2','TB.TotNbrDl.1.Cell3','TB.TotNbrDl.1.Cell4',
    #              'TB.TotNbrDl.1.Cell5','TB.TotNbrDl.1.Cell6','TB.TotNbrDl.1.Cell7','TB.TotNbrDl.1.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDl.1.Cell2','TB.TotNbrDl.1.Cell3','TB.TotNbrDl.1.Cell4',
    #              'TB.TotNbrDl.1.Cell5','TB.TotNbrDl.1.Cell6','TB.TotNbrDl.1.Cell7','TB.TotNbrDl.1.Cell8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','TB.TotNbrDl.1.Cell2','TB.TotNbrDl.1.Cell3','TB.TotNbrDl.1.Cell4',
    #              'TB.TotNbrDl.1.Cell5','TB.TotNbrDl.1.Cell6','TB.TotNbrDl.1.Cell7','TB.TotNbrDl.1.Cell8']]
    #
    #     rx_df.loc[:,['Time','TB.TotNbrDl.1.Cell2','TB.TotNbrDl.1.Cell3','TB.TotNbrDl.1.Cell4',
    #              'TB.TotNbrDl.1.Cell5','TB.TotNbrDl.1.Cell6','TB.TotNbrDl.1.Cell7','TB.TotNbrDl.1.Cell8']] = \
    #             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitial(rx_DL):
    #     rx_df = rx_DL[rx_DL['rv']==0].groupby(['cellId','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','TB.TotNbrDlInitial.Cell2','TB.TotNbrDlInitial.Cell3',
    #                                             'TB.TotNbrDlInitial.Cell4','TB.TotNbrDlInitial.Cell5',
    #                                             'TB.TotNbrDlInitial.Cell6','TB.TotNbrDlInitial.Cell7',
    #                                             'TB.TotNbrDlInitial.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDlInitial.Cell2','TB.TotNbrDlInitial.Cell3',
    #                                             'TB.TotNbrDlInitial.Cell4','TB.TotNbrDlInitial.Cell5',
    #                                             'TB.TotNbrDlInitial.Cell6','TB.TotNbrDlInitial.Cell7',
    #                                             'TB.TotNbrDlInitial.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','TB.TotNbrDlInitial.Cell2','TB.TotNbrDlInitial.Cell3',
    #                                             'TB.TotNbrDlInitial.Cell4','TB.TotNbrDlInitial.Cell5',
    #                                             'TB.TotNbrDlInitial.Cell6','TB.TotNbrDlInitial.Cell7',
    #                                             'TB.TotNbrDlInitial.Cell8']]
    #
    #     rx_df.loc[:,['Time','TB.TotNbrDlInitial.Cell2','TB.TotNbrDlInitial.Cell3',
    #                                             'TB.TotNbrDlInitial.Cell4','TB.TotNbrDlInitial.Cell5',
    #                                             'TB.TotNbrDlInitial.Cell6','TB.TotNbrDlInitial.Cell7',
    #                                             'TB.TotNbrDlInitial.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitial16Qam(rx_DL):
    #     rx_df = rx_DL[rx_DL['mcs'].isin([10,11,12,13,14,15,16])].groupby(['cellId','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','TB.TotNbrDlInitial.16Qam.Cell2','TB.TotNbrDlInitial.16Qam.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell4','TB.TotNbrDlInitial.16Qam.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell6','TB.TotNbrDlInitial.16Qam.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDlInitial.16Qam.Cell2','TB.TotNbrDlInitial.16Qam.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell4','TB.TotNbrDlInitial.16Qam.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell6','TB.TotNbrDlInitial.16Qam.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','TB.TotNbrDlInitial.16Qam.Cell2','TB.TotNbrDlInitial.16Qam.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell4','TB.TotNbrDlInitial.16Qam.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell6','TB.TotNbrDlInitial.16Qam.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell8']]
    #
    #     rx_df.loc[:,['Time','TB.TotNbrDlInitial.16Qam.Cell2','TB.TotNbrDlInitial.16Qam.Cell3',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell4','TB.TotNbrDlInitial.16Qam.Cell5',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell6','TB.TotNbrDlInitial.16Qam.Cell7',
    #                                             'TB.TotNbrDlInitial.16Qam.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitial64Qam(rx_DL):
    #     rx_df = rx_DL[rx_DL['mcs'].isin([17,18,19,20,21,22,23,24,25,26,27,28])].groupby(['cellId','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','TB.TotNbrDlInitial.64Qam.Cell2','TB.TotNbrDlInitial.64Qam.Cell3','TB.TotNbrDlInitial.64Qam.Cell4',
    #                  'TB.TotNbrDlInitial.64Qam.Cell5','TB.TotNbrDlInitial.64Qam.Cell6','TB.TotNbrDlInitial.64Qam.Cell7',
    #                  'TB.TotNbrDlInitial.64Qam.Cell8'])
    #     rx_df.loc[:,['TB.TotNbrDlInitial.64Qam.Cell2','TB.TotNbrDlInitial.64Qam.Cell3','TB.TotNbrDlInitial.64Qam.Cell4',
    #                  'TB.TotNbrDlInitial.64Qam.Cell5','TB.TotNbrDlInitial.64Qam.Cell6','TB.TotNbrDlInitial.64Qam.Cell7',
    #                  'TB.TotNbrDlInitial.64Qam.Cell8']] = list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','TB.TotNbrDlInitial.64Qam.Cell2','TB.TotNbrDlInitial.64Qam.Cell3','TB.TotNbrDlInitial.64Qam.Cell4',
    #                  'TB.TotNbrDlInitial.64Qam.Cell5','TB.TotNbrDlInitial.64Qam.Cell6','TB.TotNbrDlInitial.64Qam.Cell7',
    #                  'TB.TotNbrDlInitial.64Qam.Cell8']]
    #
    #     rx_df.loc[:,['Time','TB.TotNbrDlInitial.64Qam.Cell2','TB.TotNbrDlInitial.64Qam.Cell3','TB.TotNbrDlInitial.64Qam.Cell4',
    #                  'TB.TotNbrDlInitial.64Qam.Cell5','TB.TotNbrDlInitial.64Qam.Cell6','TB.TotNbrDlInitial.64Qam.Cell7',
    #                  'TB.TotNbrDlInitial.64Qam.Cell8']] = list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def TBTotNbrDlInitialQpsk(rx_DL):
    #     rx_df = rx_DL[rx_DL['mcs'].isin([0,1,2,3,4,5,6,7,8,9])].                                         groupby(['cellId','Time']).count().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','TB.TotNbrDlInitial.Qpsk.Cell2','TB.TotNbrDlInitial.Qpsk.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell4','TB.TotNbrDlInitial.Qpsk.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell6','TB.TotNbrDlInitial.Qpsk.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell8'])
    #
    #     rx_df.loc[:,['TB.TotNbrDlInitial.Qpsk.Cell2','TB.TotNbrDlInitial.Qpsk.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell4','TB.TotNbrDlInitial.Qpsk.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell6','TB.TotNbrDlInitial.Qpsk.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell8']] = \
    #                                              list(rx_df.loc[:,].apply(lambda x:optimize(x),axis=1))
    #
    #     rx_df = rx_df[['Time','TB.TotNbrDlInitial.Qpsk.Cell2','TB.TotNbrDlInitial.Qpsk.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell4','TB.TotNbrDlInitial.Qpsk.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell6','TB.TotNbrDlInitial.Qpsk.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell8']]
    #
    #     rx_df.loc[:,['Time','TB.TotNbrDlInitial.Qpsk.Cell2','TB.TotNbrDlInitial.Qpsk.Cell3',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell4','TB.TotNbrDlInitial.Qpsk.Cell5',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell6','TB.TotNbrDlInitial.Qpsk.Cell7',
    #                                             'TB.TotNbrDlInitial.Qpsk.Cell8']] = \
    #                                             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def RRUPrbUsedDl(RxPacketTrace):
    #     #Periodicity = 10*4
    #     #DR = Periodicity*14
    #     rx_df = RxPacketTrace.groupby(['cellId','Time']).sum().reset_index()
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','RRU.PrbUsedDl.Cell2','RRU.PrbUsedDl.Cell3',
    #                                             'RRU.PrbUsedDl.Cell4','RRU.PrbUsedDl.Cell5',
    #                                             'RRU.PrbUsedDl.Cell6','RRU.PrbUsedDl.Cell7',
    #                                             'RRU.PrbUsedDl.Cell8'])
    #
    #     rx_df.loc[:,['RRU.PrbUsedDl.Cell2','RRU.PrbUsedDl.Cell3',
    #                                             'RRU.PrbUsedDl.Cell4','RRU.PrbUsedDl.Cell5',
    #                                             'RRU.PrbUsedDl.Cell6','RRU.PrbUsedDl.Cell7',
    #                                             'RRU.PrbUsedDl.Cell8']] = \
    #              list(rx_df.loc[:,].apply(lambda x:optimize_UE_sp_PRB_schedule(x),axis=1))
    #
    #     rx_df = rx_df[['Time','RRU.PrbUsedDl.Cell2','RRU.PrbUsedDl.Cell3',
    #                                             'RRU.PrbUsedDl.Cell4','RRU.PrbUsedDl.Cell5',
    #                                             'RRU.PrbUsedDl.Cell6','RRU.PrbUsedDl.Cell7',
    #                                             'RRU.PrbUsedDl.Cell8']]
    #
    #     rx_df.loc[:,['Time','RRU.PrbUsedDl.Cell2','RRU.PrbUsedDl.Cell3',
    #                                             'RRU.PrbUsedDl.Cell4','RRU.PrbUsedDl.Cell5',
    #                                             'RRU.PrbUsedDl.Cell6','RRU.PrbUsedDl.Cell7',
    #                                             'RRU.PrbUsedDl.Cell8']] = \
    #             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def DRBMeanActiveUeDl(RxPacketTrace,UeHandoverEndStats):
    #     #to be optimized
    #     df = pd.DataFrame(columns = ['Time','DRB.MeanActiveUeDl.Cell2','DRB.MeanActiveUeDl.Cell3',
    #                           'DRB.MeanActiveUeDl.Cell4','DRB.MeanActiveUeDl.Cell5',
    #                           'DRB.MeanActiveUeDl.Cell6','DRB.MeanActiveUeDl.Cell7',
    #                           'DRB.MeanActiveUeDl.Cell8'])
    #     time_window =0.01
    #     times = np.arange(0.01, 6.001,time_window)
    #     for time in times:
    #         dt = {
    #             'Time':round(time,2),
    #             'DRB.MeanActiveUeDl.Cell2':0,'DRB.MeanActiveUeDl.Cell3':0,
    #             'DRB.MeanActiveUeDl.Cell4':0,'DRB.MeanActiveUeDl.Cell5':0,
    #             'DRB.MeanActiveUeDl.Cell6':0,'DRB.MeanActiveUeDl.Cell7':0,
    #             'DRB.MeanActiveUeDl.Cell8':0
    #             }
    #         for cell in range(2,9):
    #             #operand A
    #             condition_A= ((RxPacketTrace['time'] >= round(time,2))
    #                     & (RxPacketTrace['time'] <= round(time+0.01,2))
    #                     & (cell == RxPacketTrace['cellId']))
    #             #operand B
    #             condition_B= ((UeHandoverEndStats['Time'] >= round(time,2))
    #                     & (UeHandoverEndStats['Time'] <= round(time+0.01,2))
    #                     & (cell == UeHandoverEndStats['TargetCellId']))
    #
    #             result_A = set(RxPacketTrace[condition_A]['IMSI'])
    #             result_B = set(UeHandoverEndStats[condition_B]['IMSI'])
    #             result = len(result_A-result_B)
    #             if result != 0:
    #                 key = f'DRB.MeanActiveUeDl.Cell{cell}'
    #                 dt[key]=result
    #         df = df.append(dt, ignore_index=True)
    #     return df
    #
    # # Node level functions
    #
    # def initiate_NODE():
    #     time_window =0.01
    #     times = np.arange(0.01, 6.001,time_window)
    #     NODE = pd.DataFrame({'Time': times})
    #     NODE['Time'] = NODE['Time'].apply(lambda x:round(x,4))
    #     return NODE
    #
    # def DRBPdcpSduDelayDl(DlRlcStats):
    #     df = pd.DataFrame(columns = ['Time','DRB.PdcpSduDelayDl.Node2','DRB.PdcpSduDelayDl.Node3','DRB.PdcpSduDelayDl.Node4',
    #                              'DRB.PdcpSduDelayDl.Node5','DRB.PdcpSduDelayDl.Node6','DRB.PdcpSduDelayDl.Node7',
    #                              'DRB.PdcpSduDelayDl.Node8'])
    #     time_window =0.01
    #     times = np.arange(0.01, 6.001,time_window)
    #     for time in times:
    #         dt = {'Time':round(time,2),'DRB.PdcpSduDelayDl.Node2':0,
    #                                'DRB.PdcpSduDelayDl.Node3':0,'DRB.PdcpSduDelayDl.Node4':0,'DRB.PdcpSduDelayDl.Node5':0,
    #                                'DRB.PdcpSduDelayDl.Node6':0,'DRB.PdcpSduDelayDl.Node7':0,'DRB.PdcpSduDelayDl.Node8':0
    #         }
    #         for cell in range(2,9):
    #             condition = ((round(time,2) == DlRlcStats['start'])
    #                     & (cell == DlRlcStats['CellId']))
    #             result = 0
    #             if not DlRlcStats[condition].empty:
    #                 result = (DlRlcStats[condition]['delay']*10000).mean()
    #             key = f'DRB.PdcpSduDelayDl.Node{cell}'
    #             dt[key]=result
    #         df = df.append(dt,ignore_index=True)
    #     return df
    #
    # def DRBUEThpDl(DlRlcStats,RxPacketTrace):
    #     df = pd.DataFrame(columns=['Time','DRB.UEThpDl.Node2','DRB.UEThpDl.Node3',
    #                             'DRB.UEThpDl.Node4','DRB.UEThpDl.Node5','DRB.UEThpDl.Node6',
    #                             'DRB.UEThpDl.Node7','DRB.UEThpDl.Node8'])
    #     time_window =0.01
    #     times = np.arange(0.01, 6.001,time_window)
    #     for time in times:
    #         dt = {
    #         'Time':round(time,2),
    #         'DRB.UEThpDl.Node2':0,'DRB.UEThpDl.Node3':0,'DRB.UEThpDl.Node4':0,
    #         'DRB.UEThpDl.Node5':0,'DRB.UEThpDl.Node6':0,'DRB.UEThpDl.Node7':0,
    #         'DRB.UEThpDl.Node8':0
    #         }
    #
    #         for cell in range(2,9):
    #             dl_condition = ((DlRlcStats['start']==round(time,2))
    #                       & (DlRlcStats['CellId']==cell))
    #
    #             rl_condition = ((RxPacketTrace['time']>=round(time,2))
    #                          & (RxPacketTrace['time']<=round(time+0.01,2))
    #                          & (RxPacketTrace['cellId']==cell))
    #             dl_res = DlRlcStats[dl_condition]*8
    #             rl_res = RxPacketTrace[rl_condition]['symbol#'].sum()*10**3
    #             if not dl_res.empty and rl_res != 0:
    #                 result = dl_res/rl_res
    #                 avg = result['TxBytes'].mean()
    #                 key = f'DRB.UEThpDl.Node{cell}'
    #                 dt[key]=avg
    #         df = df.append(dt, ignore_index=True)
    #     return df
    #
    # def QosFlowPdcpPduVolumeDL_Filter(DlRlcStats):
    #     rx_df = DlRlcStats[['start','CellId','TxBytes']]
    #     rx_df['TxBytes'] = (rx_df['TxBytes']*8)/10**3
    #     rx_df = rx_df.groupby(['start','CellId']).sum().reset_index()
    #     rx_df.rename(columns={'start':'Time','TxBytes':'QosFlow.PdcpPduVolumeDL_Filter.UEID'},inplace=True)
    #     if rx_df.empty:
    #         return pd.DataFrame(columns=['Time','QosFlow.PdcpPduVolumeDL_Filter.Node2','QosFlow.PdcpPduVolumeDL_Filter.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node4','QosFlow.PdcpPduVolumeDL_Filter.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node6','QosFlow.PdcpPduVolumeDL_Filter.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node8'])
    #
    #     rx_df.loc[:,['QosFlow.PdcpPduVolumeDL_Filter.Node2','QosFlow.PdcpPduVolumeDL_Filter.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node4','QosFlow.PdcpPduVolumeDL_Filter.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node6','QosFlow.PdcpPduVolumeDL_Filter.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node8']] = \
    #                              list(rx_df.loc[:,].apply(lambda x:optimize_PdcpPduVolumeDL_Filter(x),axis=1))
    #
    #     rx_df = rx_df[['Time','QosFlow.PdcpPduVolumeDL_Filter.Node2','QosFlow.PdcpPduVolumeDL_Filter.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node4','QosFlow.PdcpPduVolumeDL_Filter.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node6','QosFlow.PdcpPduVolumeDL_Filter.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node8']]
    #
    #     rx_df.loc[:,['Time','QosFlow.PdcpPduVolumeDL_Filter.Node2','QosFlow.PdcpPduVolumeDL_Filter.Node3',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node4','QosFlow.PdcpPduVolumeDL_Filter.Node5',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node6','QosFlow.PdcpPduVolumeDL_Filter.Node7',
    #                             'QosFlow.PdcpPduVolumeDL_Filter.Node8']] = \
    #                             list(rx_df.loc[:,].apply(lambda x:max_val_cell(x,rx_df),axis=1))
    #
    #     rx_df.drop_duplicates(inplace=True)
    #     return rx_df
    #
    # def get_new_serving_cell(startStats, endStats):
    #     def func(row):
    #         condition = ((row['Time'] >= endStats['Time'])
    #                      & (row['IMSI'] == endStats['IMSI']))
    #         endStats_subset = endStats[condition]
    #         if endStats_subset.shape[0] != 0:
    #             return endStats_subset.iloc[-1, 3]
    #         else:
    #             return startStats.iloc[0, 3]
    #     return func
    #
    # def TBTotNbrDlInitial16Qam_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell2']
    #     elif row['Serving_NR'] == 3:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell3']
    #     elif row['Serving_NR'] == 4:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell4']
    #     elif row['Serving_NR'] == 5:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell5']
    #     elif row['Serving_NR'] == 6:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell6']
    #     elif row['Serving_NR'] == 7:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell7']
    #     elif row['Serving_NR'] == 8:
    #         return row['TB.TotNbrDlInitial.16Qam.Cell8']
    #
    # def TBTotNbrDlInitial64Qam_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell2']
    #     elif row['Serving_NR'] == 3:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell3']
    #     elif row['Serving_NR'] == 4:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell4']
    #     elif row['Serving_NR'] == 5:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell5']
    #     elif row['Serving_NR'] == 6:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell6']
    #     elif row['Serving_NR'] == 7:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell7']
    #     elif row['Serving_NR'] == 8:
    #         return row['TB.TotNbrDlInitial.64Qam.Cell8']
    #
    # def TBTotNbrDlInitialQpsk_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell2']
    #     elif row['Serving_NR'] == 3:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell3']
    #     elif row['Serving_NR'] == 4:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell4']
    #     elif row['Serving_NR'] == 5:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell5']
    #     elif row['Serving_NR'] == 6:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell6']
    #     elif row['Serving_NR'] == 7:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell7']
    #     elif row['Serving_NR'] == 8:
    #         return row['TB.TotNbrDlInitial.Qpsk.Cell8']
    #
    # def RRUPrbUsedDl_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['RRU.PrbUsedDl.Cell2']
    #     elif row['Serving_NR'] == 3:
    #         return row['RRU.PrbUsedDl.Cell3']
    #     elif row['Serving_NR'] == 4:
    #         return row['RRU.PrbUsedDl.Cell4']
    #     elif row['Serving_NR'] == 5:
    #         return row['RRU.PrbUsedDl.Cell5']
    #     elif row['Serving_NR'] == 6:
    #         return row['RRU.PrbUsedDl.Cell6']
    #     elif row['Serving_NR'] == 7:
    #         return row['RRU.PrbUsedDl.Cell7']
    #     elif row['Serving_NR'] == 8:
    #         return row['RRU.PrbUsedDl.Cell8']
    #
    # def DRBPdcpSduDelayDl_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['DRB.PdcpSduDelayDl.Node2']
    #     elif row['Serving_NR'] == 3:
    #         return row['DRB.PdcpSduDelayDl.Node3']
    #     elif row['Serving_NR'] == 4:
    #         return row['DRB.PdcpSduDelayDl.Node4']
    #     elif row['Serving_NR'] == 5:
    #         return row['DRB.PdcpSduDelayDl.Node5']
    #     elif row['Serving_NR'] == 6:
    #         return row['DRB.PdcpSduDelayDl.Node6']
    #     elif row['Serving_NR'] == 7:
    #         return row['DRB.PdcpSduDelayDl.Node7']
    #     elif row['Serving_NR'] == 8:
    #         return row['DRB.PdcpSduDelayDl.Node8']
    #
    # def QosFlowPdcpPduVolumeDL_Filter_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node2']
    #     elif row['Serving_NR'] == 3:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node3']
    #     elif row['Serving_NR'] == 4:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node4']
    #     elif row['Serving_NR'] == 5:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node5']
    #     elif row['Serving_NR'] == 6:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node6']
    #     elif row['Serving_NR'] == 7:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node7']
    #     elif row['Serving_NR'] == 8:
    #         return row['QosFlow.PdcpPduVolumeDL_Filter.Node8']
    #
    # #DRB.MeanActiveUeDl
    # def DRBMeanActiveUeDl_regression(row):
    #     if row['Serving_NR'] == 2:
    #         return row['DRB.MeanActiveUeDl.Cell2']
    #     elif row['Serving_NR'] == 3:
    #         return row['DRB.MeanActiveUeDl.Cell3']
    #     elif row['Serving_NR'] == 4:
    #         return row['DRB.MeanActiveUeDl.Cell4']
    #     elif row['Serving_NR'] == 5:
    #         return row['DRB.MeanActiveUeDl.Cell5']
    #     elif row['Serving_NR'] == 6:
    #         return row['DRB.MeanActiveUeDl.Cell6']
    #     elif row['Serving_NR'] == 7:
    #         return row['DRB.MeanActiveUeDl.Cell7']
    #     elif row['Serving_NR'] == 8:
    #         return row['DRB.MeanActiveUeDl.Cell8']
    #
    # def process_dir(directory):
    #     #cur_proc = mp.current_process()
    #     start_time = datetime.now().strftime("%H:%M:%S")
    #     #print(f'{cur_proc._identity}: ********* {directory} Started @ {start_time} ************')
    #     cwd = os.getcwd()
    #     data_dir = os.path.join(cwd, directory)
    #
    #     # skip dirs that were already processed
    # 	#Add regression PM's
    #     if os.path.exists(os.path.join(data_dir, 'final_df.csv')):
    #         processed_df = pd.read_csv(os.path.join(data_dir, 'final_df.csv'))
    #         if 'Serving_NR' in processed_df.columns:
    #             del processed_df
    #             #print(f'{cur_proc._identity}: --------- {directory} already processed - skipping ---------')
    #             return
    #         else:
    #             #print(f'{cur_proc._identity}: --------- {directory} including Serving_NR ---------')
    #             #sort the df
    #             processed_df = processed_df.sort_values(['IMSI','Time'])
    #             #import handoverstartstat & endstat
    #             file = 'UeHandoverStartStats.txt'
    #             UeHandoverStartStats = pd.read_csv(os.path.join(data_dir,file), sep=' ',
    #                                        names=['time', 'IMSI', 'RNTI', 'SourceCellId', 'TargetCellId'])
    #             file = 'UeHandoverEndStats.txt'
    #             UeHandoverEndStats = pd.read_csv(os.path.join(data_dir,file), sep=' ',
    #                                         names=['time','IMSI','RNTI','Serving_NR'])
    #             #Create time into ms window
    #             UeHandoverEndStats['Time'] = UeHandoverEndStats['time'].apply(lambda x:float(str(x)[:4]))
    #             #remove duplicate rows keep thelatest record.
    #             UeHandoverEndStats = UeHandoverEndStats[~UeHandoverEndStats.duplicated(['IMSI','Time'],keep='last')]
    #             #Create Serving_NR
    #             processed_df = pd.merge(processed_df,UeHandoverEndStats[['Time','IMSI','Serving_NR']],how='left',on=['Time','IMSI'])
    #
    # 		    #Forward fill
    #             for imsi in processed_df['IMSI'].unique():
    #                 processed_df.loc[processed_df['IMSI']==imsi,'Serving_NR'] = processed_df.loc[processed_df['IMSI']==imsi,'Serving_NR'].fillna(method='ffill')
    #                 #extract the first record time for that IMSI HO
    #                 time = UeHandoverEndStats[UeHandoverEndStats['IMSI']==imsi].sort_values('time').iloc[0,:]['time']
    #                 #extract the source cell id from uehandoverstarts for that time
    #                 cell = UeHandoverStartStats[(UeHandoverStartStats['time']<=time) & (UeHandoverStartStats['IMSI']==imsi)].sort_values('time').iloc[-1,:]['SourceCellId']
    #                 processed_df.loc[processed_df['IMSI']==imsi,'Serving_NR'] = processed_df.loc[processed_df['IMSI']==imsi,'Serving_NR'].fillna(cell)
    #
    #             processed_df['TB.TotNbrDlInitial.16Qam'] = processed_df.apply(lambda x: TBTotNbrDlInitial16Qam_regression(x),axis=1)
    #             processed_df['TB.TotNbrDlInitial.64Qam'] = processed_df.apply(lambda x: TBTotNbrDlInitial64Qam_regression(x),axis=1)
    #             processed_df['TB.TotNbrDlInitial.Qpsk'] = processed_df.apply(lambda x: TBTotNbrDlInitialQpsk_regression(x),axis=1)
    #             processed_df['RRU.PrbUsedDl'] = processed_df.apply(lambda x: RRUPrbUsedDl_regression(x),axis=1)
    #             processed_df['DRB.PdcpSduDelayDl'] = processed_df.apply(lambda x: DRBPdcpSduDelayDl_regression(x),axis=1)
    #             processed_df['QosFlow.PdcpPduVolumeDL_Filter'] = processed_df.apply(lambda x: QosFlowPdcpPduVolumeDL_Filter_regression(x),axis=1)
    #             processed_df['DRB.MeanActiveUeDl'] = processed_df.apply(lambda x: DRBMeanActiveUeDl_regression(x),axis=1)
    #
    # 			#filter IMSI having more than 2 HO
    #             processed_df['Previous_NR'] = processed_df.groupby('IMSI').agg(HO =('Serving_NR','shift'))
    #             #Find the HO cnts
    #             IMSI_HO_df = processed_df[(processed_df['Serving_NR']!=processed_df['Previous_NR'])].groupby(['IMSI']).count().reset_index()[['IMSI','Previous_NR']]
    #             #Filter the UE's with more than 2 HO
    #             IMSI_HO = IMSI_HO_df[IMSI_HO_df['Previous_NR']>=2]['IMSI']
    #
    #             processed_df[processed_df['IMSI'].isin(IMSI_HO)][['Time','IMSI','Serving_NR',
    #                           'DRB.PdcpSduDelayDl.UEID','Tot.PdcpSduNbrDl.UEID','DRB.PdcpSduBitRateDl.UEID','TB.TotNbrDlInitial.16Qam',
    # 						  'TB.TotNbrDlInitial.64Qam','TB.TotNbrDlInitial.Qpsk','RRU.PrbUsedDl','DRB.PdcpSduDelayDl',
    # 						  'QosFlow.PdcpPduVolumeDL_Filter','DRB.MeanActiveUeDl' ]].to_csv(os.path.join(data_dir, 'Regression_PM.csv'), index=False)
    #             end_time = datetime.now().strftime("%H:%M:%S")
    #             del IMSI_HO,IMSI_HO_df,processed_df,time,cell,UeHandoverStartStats,UeHandoverEndStats
    #             #print(f'{cur_proc._identity}: ********* {directory} Completed @ {end_time}  ************')
    #             return
    #
    #
    #
    #
    #
    # 		#if 'new_current_serving_NR' in processed_df.columns:
    #         #    del processed_df
    #         #    print(f'{cur_proc._identity}: --------- {directory} already processed - skipping ---------')
    #         #    return
    #         #else:
    #         #    print(f'{cur_proc._identity}: --------- {directory} has final_df - including new column ---------')
    #         #    UeHandoverStartStats_ = pd.read_csv(os.path.join(data_dir, 'UeHandoverStartStats.txt'),
    #         #                                       sep=' ',
    #         #                                       names=['Time', 'IMSI', 'RNTI', 'SourceCellId', 'TargetCellId'])
    #         #    UeHandoverEndStats_ = pd.read_csv(os.path.join(data_dir, 'UeHandoverEndStats.txt'),
    #         #                                     sep=' ',
    #         #                                     names=['Time','IMSI','RNTI','TargetCellId'])
    #         #    processed_df['new_current_serving_NR'] = processed_df.agg(get_new_serving_cell(UeHandoverStartStats_, UeHandoverEndStats_), axis=1)
    #         #    processed_df.to_csv(os.path.join(data_dir, 'final_df.csv'), index=False)
    #         #    del UeHandoverStartStats_, UeHandoverEndStats_
    #         #    end_time = datetime.now().strftime("%H:%M:%S")
    #         #    print(f'{cur_proc._identity}: ********* {directory} Completed @ {end_time}  ************')
    #         #    return
    #
    #
    #
    #
    #     #select .txt files
    #     #file_list = [file for file in os.listdir(data_dir) if file.endswith('.txt')]
    #     #
    #     ## skip if no txt files are present
    #     #if len(file_list) == 0:
    #     #    print(f'{cur_proc._identity}: --------- {directory} does not have any text files - skipping...')
    #     #    return
    #     #
    #     #global DlPcdpStats, DlRlcStats, UeHandoverStartStats, UeHandoverEndStats, MmWaveSineTime, RxPacketTrace
    #     #
    #     ##create dataframes
    #     #for file in file_list:
    #     #    if file == 'DlPdcpStats.txt':
    #     #        DlPdcpStats = pd.read_csv(os.path.join(data_dir,file), sep='\t', index_col=False)
    #     #        DlPdcpStats.rename(columns={'% start': 'start'}, inplace=True)
    #     #        DlPdcpStats.drop(columns=['stdDev', 'min', 'max', 'stdDev.1', 'min.1', 'max.1'], inplace=True)
    #     #    elif file == 'DlRlcStats.txt':
    #     #        DlRlcStats = pd.read_csv(os.path.join(data_dir,file), sep='\t', index_col=False)
    #     #        DlRlcStats.rename(columns={'% start': 'start'}, inplace=True)
    #     #        DlRlcStats.drop(columns=['stdDev', 'min', 'max', 'stdDev.1', 'min.1', 'max.1'], inplace=True)
    #     #    elif file == 'UeHandoverStartStats.txt':
    #     #        UeHandoverStartStats = pd.read_csv(os.path.join(data_dir,file), sep=' ',
    #     #                        names=['Time', 'IMSI', 'RNTI', 'SourceCellId', 'TargetCellId'])
    #     #    elif file == 'UeHandoverEndStats.txt':
    #     #        UeHandoverEndStats = pd.read_csv(os.path.join(data_dir,file), sep=' ',
    #     #                                         names=['Time','IMSI','RNTI','TargetCellId'])
    #     #    elif file == 'MmWaveSinrTime.txt':
    #     #        MmWaveSinrTime = pd.read_csv(os.path.join(data_dir,file), sep = ' ',
    #     #                                     names = ['Time', 'IMSI', 'CellId', 'SINR'])
    #     #    elif file == 'RxPacketTrace.txt':
    #     #        RxPacketTrace = pd.read_csv(os.path.join(data_dir,file), sep='\t', index_col=False)
    #     #RxPacketTrace['IMSI'] = RxPacketTrace.agg(getIMSI, axis=1)
    #     ##drop nulls
    #     #RxPacketTrace.drop(RxPacketTrace[RxPacketTrace['IMSI'].isnull()].index, inplace=True)
    #     #RxPacketTrace['IMSI'] = RxPacketTrace['IMSI'].astype('int32')
    #     #RxPacketTrace['Time'] = RxPacketTrace['time'].apply(lambda x:float(str(x)[:4]))
    #     #RxPacketTrace_DL = RxPacketTrace[(RxPacketTrace['DL/UL']=='DL') & (RxPacketTrace['cellId']!=1)]
    #     #RxPacketTrace['time_cellid_imsi'] =RxPacketTrace.apply(lambda x: (x['Time'],x['cellId'],x['IMSI']),axis=1)
    #     #RxPacketTrace['time_cellid'] =RxPacketTrace.apply(lambda x: (x['Time'],x['cellId']),axis=1)
    #     ##***************************UE level processing*****************************************#
    #     ##initiate UE & rx_sinr_mcs_df
    #     #UE = initial_UE()
    #     ##rx_sinr_mcs_df = initial_rx_sinr_mcs()
    #     ##L1M.RS-SINR.BinX.UEID
    #     ##CARR.PDSCHMCSDist.BinX.BinY.BinZ.UEID
    #     ##UE = get_SINR_MCS_UEID(UE,rx_sinr_mcs_df)
    #     #UE = pd.merge(UE,get_SINR_MCS_UEID(RxPacketTrace),on=['Time','IMSI'],how='left').fillna(0)
    #     ##UE_sinr_mcs_bkp = UE.copy()
    #     ##TB.TotNbrDl.1.UEID
    #     #UE = pd.merge(UE,TBTotNbrDl1UEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.UEID
    #     #UE = pd.merge(UE,TBTotNbrDlInitialUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##DRB.PdcpBitrate.QOS.UEID
    #     ##UE = pd.merge(UE,DRBPdcpBitrateQOSUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##TB.InitialErrNbrDl.UEID
    #     #UE = pd.merge(UE,TBInitialErrNbrDlUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.Qpsk.UEID
    #     #UE = pd.merge(UE,TBTotNbrDlInitialQpskUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.16Qam.UEID
    #     #UE = pd.merge(UE,TBTotNbrDlInitial16QamUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.64Qam.UEID
    #     #UE = pd.merge(UE,TBTotNbrDlInitial64QamUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     ##Tot.PdcpSduNbrDl.UEID,
    #     ##DRB.PdcpSduDelayDl.UEID,
    #     ##DRB.PdcpSduBitRateDl.UEID,
    #     ##DRB.PdcpSduVolumeDl_Filter.UEID
    #     #UE = pd.merge(UE,UELteUEID(DlPdcpStats),on=['Time','IMSI'],how='left').fillna(0)
    #     #
    #     ##DRB.IPTimeDL.QOS.UEID
    #     #UE = pd.merge(UE,DRBIPTimeDLQOSUEID(RxPacketTrace_DL),on=['Time','IMSI'],how='left').fillna(0)
    #     #
    #     ##current_serving_nr
    #     #rx_df = RxPacketTrace.groupby(['Time','IMSI']).max().reset_index()[['Time','IMSI','time']]
    #     #rx_df.loc[:,'current_serving_NR'] = rx_df.loc[:,].apply(lambda x: get_serving_cell(x),axis=1)
    #     #rx_df.drop('time',axis=1,inplace=True)
    #     #UE = pd.merge(UE,rx_df,on=['Time','IMSI'],how='left')
    #     ##forward fill the missing values
    #     #for imsi in range(1,50):
    #     #    UE.loc[UE['IMSI']==imsi,'current_serving_NR'] = UE.loc[UE['IMSI']==imsi,'current_serving_NR'].fillna(method='ffill')
    #     #    UE['current_serving_NR'].fillna(8,inplace=True)
    #     ##RRU.PrbUsedDl.UEID
    #     #UE = pd.merge(UE,RRUPrbUsedDlUEID(RxPacketTrace),on=['Time','IMSI'],how='left').fillna(0)
    #     ##DRB.PdcpPduNbrDl.Qos.UEID
    #     #UE = pd.merge(UE,DRBPdcpPduNbrDlQosUEID(DlRlcStats),on=['Time','IMSI'],how='left').fillna(np.nan)
    #     ##QosFlow.PdcpPduVolumeDL_Filter.UEID
    #     #UE = pd.merge(UE,QosFlowPdcpPduVolumeDL_FilterUEID(DlRlcStats),on=['Time','IMSI'],how='left').fillna(np.nan)
    #     #
    #     ##***************************CELL level processing*****************************************#
    #     #
    #     #CELL = initiate_CELL()
    #     ##rx_sinr_mcs_cell_df = initial_rx_sinr_mcs_cell(RxPacketTrace)
    #     ##CARR.PDSCHMCSDist.BinX.BinY.BinZ,L1M.RS-SINR.BinX
    #     ##CELL = get_SINR_MCS_CELL(CELL,rx_sinr_mcs_cell_df)
    #     #CELL = pd.merge(CELL,get_SINR_MCS_CELL(RxPacketTrace),on='Time',how='left').fillna(0)
    #     ##TB.InitialErrNbrDl
    #     #CELL = pd.merge(CELL,TBInitialErrNbrDl(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    #     ##TB.TotNbrDl.1
    #     #CELL = pd.merge(CELL,TBTotNbrDl1(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    #     ##TB.TotNbrDlInitial
    #     #CELL = pd.merge(CELL,TBTotNbrDlInitial(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.16Qam
    #     #CELL = pd.merge(CELL,TBTotNbrDlInitial16Qam(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.64Qam
    #     #CELL = pd.merge(CELL,TBTotNbrDlInitial64Qam(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    #     ##TB.TotNbrDlInitial.Qpsk
    #     #CELL = pd.merge(CELL,TBTotNbrDlInitialQpsk(RxPacketTrace_DL),on='Time',how='left').fillna(0)
    #     ##RRU.PrbUsedDl
    #     #CELL = pd.merge(CELL,RRUPrbUsedDl(RxPacketTrace),on='Time',how='left').fillna(0)
    #     ##DRB.MeanActiveUeDl
    #     #CELL = pd.merge(CELL,DRBMeanActiveUeDl(RxPacketTrace,UeHandoverEndStats),on='Time',how='left')
    #     ##DL_TotalofAvailablePRBs
    #     #CELL['DL_TotalofAvailablePRBs'] = 139
    #     #
    #     ##******************************Node level processing*************************#
    #     #NODE = initiate_NODE()
    #     ##DRB.PdcpSduDelayDl
    #     #NODE = DRBPdcpSduDelayDl(DlRlcStats)
    #     ##DRB.UEThpDl
    #     #NODE = pd.merge(NODE,DRBUEThpDl(DlRlcStats,RxPacketTrace),on='Time')
    #     ##QosFlow.PdcpPduVolumeDL_Filter
    #     #NODE = pd.merge(NODE,QosFlowPdcpPduVolumeDL_Filter(DlRlcStats),on='Time',how='left')
    #     ###*************************** final df**************************************#
    #     #final_df = pd.merge(pd.merge(UE,CELL,on='Time',how='left'),NODE,on='Time',how='left')
    #     #final_df['new_current_serving_NR'] = final_df.agg(get_new_serving_cell(UeHandoverStartStats, UeHandoverEndStats)) # new serving cell column
    #     #final_df.to_csv(os.path.join(data_dir,'final_df.csv'), index=False)
    #     #
    #     ##*******Remove temp variables**************#
    #     #del NODE,CELL,UE,DlPdcpStats,DlRlcStats,UeHandoverStartStats,UeHandoverEndStats,MmWaveSinrTime,RxPacketTrace,RxPacketTrace_DL
    #     #end_time = datetime.now().strftime("%H:%M:%S")
    #     #print(f'{cur_proc._identity}: *********{directory} Completed @ {end_time}  ************')
    #
    # if __name__ == '__main__':
    #     #if len(sys.argv) > 1:
    #     #    num_procs = int(sys.argv[1])
    #     #else:
    #     #    num_procs = mp.cpu_count() - 1
    #     dirs = [d for d in glob.glob('*') if os.path.isdir(d)]
    #     #print(dirs)
    #     process_dir(dirs[0])
    #     #with mp.Pool(processes=num_procs) as process_pool:
    #      #   process_pool.map(process_dir, dirs)pu_count() - 1
    dirs = [d for d in glob.glob('*') if os.path.isdir(d)]
    # print(dirs)
    process_dir(dirs[0])
    # with mp.Pool(processes=num_procs) as process_pool:
    #   process_pool.map(process_dir, dirs)
