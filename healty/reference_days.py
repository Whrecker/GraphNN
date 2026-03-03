import pandas as pd
import random as r
#add location
#add frequncy for bathroom, sleep breaks
#priority
data = {"activity": [],"location":[], "start time": [], "end time": [], "duration": []}

def assign(a, s, d,l):
    data["activity"].append(a)
    data["start time"].append(s)
    data["end time"].append(s + d)
    data["duration"].append(d)
    data["location"].append(l)
def daily():
    de = r.randrange(0, 20)
    events = {
        "1st": {
            ("waking up","bedroom"): (float('inf'), 0),
            ("exercising","bedroom"): (r.randrange(-5, 10), r.randrange(15, 45)),
            ("taking medicine","bedroom"): (r.randrange(-1, 100), 1),
            ("bathing","bathroom"): (r.randrange(-5, 100), r.randrange(5, 15)),
            ("brushing","bathroom"): (r.randrange(-1, 200), r.randrange(1, 3)),
            ("bathroom_excretion","bathroom"): (r.randrange(1, 100), r.randrange(3, 10)),
            ("watching tv","living room"): (0, 0)
        },
        "2nd": {
            ("making food","kitchen"): (float('inf'), r.randrange(15, 45)),
            ("eating","dining table"): (float('inf'), r.randrange(5, 20)),
            ("bathroom1","bathroom"): (r.randrange(-10, 50), 1),
            ("watching tv2","living room"): (0, 0)
        },
        "3rd": {
            ("making food2","kitchen"): (float('inf'), r.randrange(15, 45)),
            ("eating2","dining table"): (float('inf'), r.randrange(5, 20)),
            ("bathroom2","bathroom"): (r.randrange(-10, 50), 1),
            ("nap","bedroom"): (float('inf'), 0)
            },
        "4th": {
            ("bathroom3","bathroom"): (r.randrange(-10, 50), 1),
            ("walk","out"): (r.randrange(-1, 1), r.randrange(90, 120)),
            ("watchingtv3","living room"): (0, 0)
        },
        "5th": {
            ("making food3","kitchen"): (float('inf'), r.randrange(15, 45)),
            ("eating3","dining table"): (r.randrange(5, 50), r.randrange(5, 20)),
            ("watching tv4","living room"): (r.randrange(-5, 5), r.randrange(60, 120)),
            ("bathroom4","bathroom"): (r.randrange(-10, 50), 1),
            ("sleep","bedroom"): (0, 1)
        }
    }

    counter = 0
    timestamps = [420, 9 * 60, 13 * 60, 17 * 60, 20 * 60]
    for i1, j1 in events.items():
        sorted_events = sorted(j1.items(), key=lambda x: x[1][0], reverse=True)
        time = timestamps[counter]
        counter += 1
        
        for i2, j2 in sorted_events:
            if i2[0] == "sleep":
                data["activity"].append(i2[0])
                data["start time"].append(time)
                data["end time"].append(7 * 60)
                data["duration"].append(30 * 60 - time)
                data["location"].append(i2[1])
                break
            elif i2[0]=="waking up":
                assign(i2[0],time,j2[1],i2[1])
            elif j2[0] >= 0:
                if j2[1] > 0:
                    assign(i2[0], time, j2[1], i2[1])
                else:
                    assign(i2[0], time, timestamps[counter] - time, i2[1])
            time += j2[1]

for i in range(100):
    print(i)
    daily()
    df=pd.DataFrame(data)
    df.to_excel(f"day{i}.xlsx",index=False)
    data = {"activity": [],"location":[], "start time": [], "end time": [], "duration": []}
