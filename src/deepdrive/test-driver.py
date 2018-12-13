import speed_dreams as sd

def initMemory():
    print("Init Memory")
    Memory = sd.CSharedMemory()
    Memory.setSyncMode(True)
    return Memory


Memory = initMemory()

# Publish the values on SHM
Memory.Data.Control.Steering = 0.0
Memory.Data.Control.Accelerating = 0.0
# Abnormal situation
Memory.Data.Control.Breaking = 3.0
print("Create Abnormal Situation")

# Notify the wrapper about it
Memory.indicateReady()

# Read Again anc check right value is there
import time

while True:
    time.sleep(0.1)
    print("Value From Memory", Memory.Data.Control.Breaking)