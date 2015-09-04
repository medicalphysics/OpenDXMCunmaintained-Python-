# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:52:33 2015

@author: erlean
"""
import math

seconds = [86400., 3600., 60., 1.]
names = ['days', 'hours', 'minutes', 'seconds']

#times ={'days': 86400.,
#        'hours': 3600.,
#        'minutes': 60}

def human_time(sec):
    if sec <= seconds[-1]:
        return 'less than a {0}'.format(names[-1][:-1])
    times = []
    labels = []
    for i, s in enumerate(seconds):
        if sec >= s:
            times.append(int(math.floor(sec/s)))
            labels.append(names[i])
            if times[0] == 1:
                labels[0] = labels[0][:-1]
            if i < len(seconds) - 1:
                times.append(int(math.floor((sec - s * times[0])/seconds[i+1])))
                labels.append(names[i+1])
                if times[1] == 1:
                    labels[1] = labels[1][:-1]

            break
    else:
        return 'less than a {0}'.format(names[-1][:-1])
    if len(times) > 1:
       if times[-1] == 0:
           return 'about {0} {1}'.format(times[0], labels[0])
       return 'about {0} {1} and {2} {3}'.format(times[0], labels[0], times[1], labels[1])
    return 'about {0} {1}'.format(times[0], labels[0])



if __name__ == '__main__':
    print((human_time(3660)))




