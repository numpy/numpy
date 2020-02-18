
def addInDict(dict, fileName):
    f = open(fileName, "r")
    for line in f:
        elements = line.replace('=', ' ').split()
        for e in elements:
            if e!="":
                dict[e]=e
    return dict

padDict={}
padDict = addInDict(padDict, "./build/testenv/lib/python3.8/site-packages/pad.txt")
padTotalBranches = 27   #need to update manually
print("coverage pad: ", len(padDict)/padTotalBranches)

arrayDict={}
arrayDict = addInDict(arrayDict, "./build/testenv/lib/python3.8/site-packages/array.txt")
arrayTotalBranches = 27   #need to update manually
print("coverage array: ", len(arrayDict)/arrayTotalBranches)

gradientDict={}
gradientDict = addInDict(gradientDict, "./build/testenv/lib/python3.8/site-packages/gradient.txt")
gradientTotalBranches = 30   #need to update manually
print("coverage gradient: ", len(gradientDict)/gradientTotalBranches)

normDict={}
normDict = addInDict(normDict, "./build/testenv/lib/python3.8/site-packages/norm.txt")
normTotalBranches = 34   #need to update manually
print("coverage norm: ", len(normDict)/normTotalBranches)

polyfitDict={}
polyfitDict = addInDict(polyfitDict, "./build/testenv/lib/python3.8/site-packages/polyfit.txt")
polyfitTotalBranches = 20   #need to update manually
print("coverage polyfit: ", len(polyfitDict)/polyfitTotalBranches)

scanDict={}
scanDict = addInDict(scanDict, "./build/testenv/lib/python3.8/site-packages/scan.txt")
scanTotalBranches = 52   #need to update manually
print("coverage scan: ", len(scanDict)/scanTotalBranches)

tensordotDict={}
tensordotDict = addInDict(tensordotDict, "./build/testenv/lib/python3.8/site-packages/tensordot.txt")
tensordotTotalBranches = 13   #need to update manually
print("coverage tensordot: ", len(tensordotDict)/tensordotTotalBranches)

'''
We don't have the file for buildcallback coverage yet...

buildcallbackDict={}
buildcallbackDict = addInDict(buildcallbackDict, "./build/testenv/lib/python3.8/site-packages/buildcallback.txt")
buildcallbackTotalBranches = 34   #need to update manually
print("coverage buildcallback: ", len(buildcallbackDict)/buildcallbackTotalBranches)
'''
