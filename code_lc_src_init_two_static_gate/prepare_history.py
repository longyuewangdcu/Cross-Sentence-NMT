#coding=utf-8
__author__ = 'vincent'

import file2context
import os.path


if __name__ == '__main__':
    work_dir='/Users/Violeta/PycharmProjects/NMT-ZPTU/data/nist/nist02/org'#+key

    # LANG = 'zh'
    # SUFFIX = 'np.tk.qb.es' #zh

    LANG = 'en3'
    SUFFIX = 'qb.np.tk.lc' #en

    # FILE_NAMES = ['.'.join(name.split('.')[:-1]) for root, dirs, files in os.walk(work_dir) for name in files if name.find('context')<0 and name.find(SUFFIX+LANG)>=0]
    # print FILE_NAMES
    # FILE_NAMES = list(set(FILE_NAMES))
    # print 'all files:', FILE_NAMES
    # MAX_HIST_LEN = 1
    #
    # for FILE_NAME in FILE_NAMES:
    #     print 'processing',FILE_NAME
    #     file2context.file2context(work_dir,FILE_NAME,LANG,MAX_HIST_LEN)

    FILE_NAMES = []
    for root, dirs, files in os.walk(work_dir):
        for name in files:
            if name.find('hist')<0 and name.find(SUFFIX+'.'+LANG)>=0:
                # FILE_NAMES.append(root+'/'+'.'.join(name.split('.')[:-1]))
                FILE_NAMES.append(int(name.split('.')[0]))

    print (FILE_NAMES)
    FILE_NAMES = sorted(FILE_NAMES)
    print (FILE_NAMES)

    # FILE_NAMES = FILE_NAMES[0:2]
    MAX_HIST_LEN = 3

    for FILE_NAME in FILE_NAMES:
        print 'processing',FILE_NAME
        file2context.file2context(work_dir + '/' + str(FILE_NAME) + '.' + SUFFIX, LANG, MAX_HIST_LEN)