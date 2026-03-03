import pandas as pd
import random as r

data = {"activity": [], "location": [], "start time": [], "end time": [], "duration": []}

def assign(a, s, d, l):
    data["activity"].append(a)
    data["start time"].append(s)
    data["end time"].append(s + d)
    data["duration"].append(d)
    data["location"].append(l)

def daily():
    de = r.randrange(0, 20)
    events = {
        "1st": {
            ("waking up", r.choice(["living room"])): (float('inf'), 0),
            ("exercising", r.choice(["living room"])): (r.randrange(-50, 10), r.randrange(1, 5)),
            ("taking medicine", r.choice(["living room"])): (r.randrange(-10, 100), 5),
            ("bathing", "bathroom"): (r.randrange(-50, 100), r.randrange(15, 60)),
            ("brushing", r.choice(["bedroom"])): (r.randrange(-1, 200), r.randrange(1, 3)),
            ("bathroom_excretion", "bathroom"): (r.randrange(1, 100), r.randrange(3, 10)),
            ("watching tv", r.choice(["bedroom", "living room"])): (0, 0)
        },
        "2nd": {
            ("making food", "kitchen"): (float('inf'), r.randrange(15, 45)),
            ("eating", r.choice(["bedroom", "living room"])): (float('inf'), r.randrange(1, 5)),
            ("bathroom1", "bathroom"): (r.randrange(-10, 50), 5),
            ("watching tv2", r.choice(["bedroom", "living room"])): (0, 0)
        },
        "3rd": {
            ("making food2", "kitchen"): (float('inf'), r.randrange(15, 45)),
            ("eating2", r.choice(["bedroom", "living room"])): (float('inf'), r.randrange(20, 50)),
            ("bathroom2", "bathroom"): (r.randrange(-10, 50), 1),
            ("nap", r.choice([ "living room"])): (float('inf'), 0)
        },
        "4th": {
            ("bathroom3", "bathroom"): (r.randrange(-10, 50), 1),
            ("walk", r.choice(["living room"])): (r.randrange(-1, 1), r.randrange(10, 15)),
            ("watchingtv3", r.choice(["bedroom", "living room"])): (0, 0)
        },
        "5th": {
            ("making food3", "kitchen"): (float('inf'), r.randrange(15, 45)),
            ("eating3", r.choice(["bedroom", "living room"])): (r.randrange(5, 50), r.randrange(15, 40)),
            ("watching tv4", r.choice(["bedroom", "living room"])): (r.randrange(-5, 5), r.randrange(60, 120)),
            ("bathroom4", "bathroom"): (r.randrange(-10, 50), 1),
            ("sleep", r.choice(["living room"])): (0, 1)
        }
    }

    counter = 0
    timestamps = [600, 12 * 60, 14 * 60, 17 * 60, 20 * 60]
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
    daily()
    df = pd.DataFrame(data)
    df.to_excel(f"day{i}.xlsx", index=False)
    data = {"activity": [], "location": [], "start time": [], "end time": [], "duration": []}
