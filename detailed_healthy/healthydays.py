import pandas as pd
import random as r

for j in range(200):
    data={
    "time(in seconds)":[],
    "bed pressure sensor":[],
    "bedroom sensor":[],
    "kitchen sensor":[],
    "couch pressure sensor":[],
    "dinning chair pressure sensor":[],
    "bathroom sensor":[],
    "house sensor":[],
    }
    print(j)
    temp=(7*3600)+r.randrange(-1800,1800,5)
    temp2=r.randrange(300,1200,5)
    temp3=r.randrange(60,120,5)
    temp4=r.randrange(900,2700,5)
    temp5=r.randrange(-1800,1800,5)
    ran1=r.random()
    ran2=r.random()
    ran3=r.random()
    ran4=r.random()
    ran5=r.random()
    for i in range(0,(24*3600)+1,5):
        data["time(in seconds)"].append(i)
        if i<=temp:
            data["bed pressure sensor"].append(1)
            data["bedroom sensor"].append(1)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        if i>temp and i<temp+temp2:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(1)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(1)
            data["house sensor"].append(1)
        if i>=temp+temp2 and i<temp+temp2+15:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(1)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        if i>=temp+temp2+15 and i<8*3600:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(1)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        if i>=(8*3600) and i<(8*3600)+temp4:        
            if ran1>.5:
                data["bed pressure sensor"].append(0)
                data["bedroom sensor"].append(0)
                data["kitchen sensor"].append(1)
                data["couch pressure sensor"].append(0)
                data["dinning chair pressure sensor"].append(0)
                data["bathroom sensor"].append(0)
                data["house sensor"].append(1)
            else:
                
                if i<(8*3600)+temp3:
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(1)
                    data["house sensor"].append(1)
                if i>=(8*3600)+temp3:
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(0)
                    data["kitchen sensor"].append(1)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(0)
                    data["house sensor"].append(1)
        
        if i>=(8*3600)+temp4 and i<(8*3600)+temp4+temp4:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(1)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        
        if i>=(8*3600)+temp4+temp4 and i<13*3600:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(1)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        
        if i>=13*3600 and i<(13*3600)+temp4:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(1)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        
        if i>=(13*3600)+temp4 and i<(13*3600)+temp4+temp4:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(1)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        
        if i>=(13*3600)+temp4+temp4 and i<18*3600+temp5:
            if ran2>.5:
                data["bed pressure sensor"].append(1)
                data["bedroom sensor"].append(1)
                data["kitchen sensor"].append(0)
                data["couch pressure sensor"].append(0)
                data["dinning chair pressure sensor"].append(0)
                data["bathroom sensor"].append(0)
                data["house sensor"].append(1)
            else:
                if i<=(13*3600)+temp4+temp4+temp3:                    
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(1)
                    data["house sensor"].append(1)
                else:                    
                    data["bed pressure sensor"].append(1)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(0)
                    data["house sensor"].append(1)
        
        if i>=18*3600+temp5 and i<20*3600:
            if ran3>.5:
                data["bed pressure sensor"].append(0)
                data["bedroom sensor"].append(0)
                data["kitchen sensor"].append(0)
                data["couch pressure sensor"].append(0)
                data["dinning chair pressure sensor"].append(0)
                data["bathroom sensor"].append(0)
                data["house sensor"].append(0)
            else:
                if i<=18*3600+temp2+temp3:                
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(1)
                    data["house sensor"].append(1)
                else:
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(0)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(0)
                    data["house sensor"].append(0)
        
        if i >=20*3600 and i<(20*3600)+temp4:
            if ran4>.5:
                data["bed pressure sensor"].append(0)
                data["bedroom sensor"].append(0)
                data["kitchen sensor"].append(1)
                data["couch pressure sensor"].append(0)
                data["dinning chair pressure sensor"].append(0)
                data["bathroom sensor"].append(0)
                data["house sensor"].append(1)
            else:
                if i<20*3600+temp3:
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(1)
                    data["house sensor"].append(1)
                else:
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(0)
                    data["kitchen sensor"].append(1)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(0)
                    data["house sensor"].append(1)
        
        if i >= (20*3600)+temp4 and i <(20*3600)+temp4+temp4:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(0)
            data["dinning chair pressure sensor"].append(1)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        
        if i>=(20*3600)+temp4+temp4 and i<22*3600:
            data["bed pressure sensor"].append(0)
            data["bedroom sensor"].append(0)
            data["kitchen sensor"].append(0)
            data["couch pressure sensor"].append(1)
            data["dinning chair pressure sensor"].append(0)
            data["bathroom sensor"].append(0)
            data["house sensor"].append(1)
        
        if i>=22*3600:
            if ran5>.5:
                data["bed pressure sensor"].append(1)
                data["bedroom sensor"].append(1)
                data["kitchen sensor"].append(0)
                data["couch pressure sensor"].append(0)
                data["dinning chair pressure sensor"].append(0)
                data["bathroom sensor"].append(0)
                data["house sensor"].append(1)
            else:
                if i<22*3600+temp3:
                    data["bed pressure sensor"].append(0)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(1)
                    data["house sensor"].append(1)
                else:
                    data["bed pressure sensor"].append(1)
                    data["bedroom sensor"].append(1)
                    data["kitchen sensor"].append(0)
                    data["couch pressure sensor"].append(0)
                    data["dinning chair pressure sensor"].append(0)
                    data["bathroom sensor"].append(0)
                    data["house sensor"].append(1)
        if len(data["bed pressure sensor"])!=len(data["time(in seconds)"]):
            print("-----------------------")
            print(i)
            print(i/5)
            print(len(data["bed pressure sensor"]))
            print(len(data["time(in seconds)"]))
            break
    df=pd.DataFrame(data)
    df.to_excel(f'healthyday{j}.xlsx',index=False)
print("done")
