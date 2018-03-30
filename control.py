import math
import time
import pypot.dynamixel
import itertools
import numpy as np
ports = pypot.dynamixel.get_available_ports()
dxl_io = pypot.dynamixel.DxlIO(ports[0])
def convertcar2pol(x,y):
    r=math.sqrt(x**2+y**2)
    theta=math.atan(y/x)*180/math.pi
    if theta<0:
        theta=theta+180
    return r,theta
def control(alpha,delta,theta,s1,s2,s3,s4,e1,e2,e3,xT,yT,l,t):

    K= np.matrix([[4,0,0,0],[0,0.4,0,0],[0,0,1,0],[0,0,0,0]])
    
    rT,thetaT=convertcar2pol(xT,yT)
    thetaT=math.pi/180
    
    if(t>100):
        t=0.02
    x = (49+(l**2)-(s4**2))/(2*7*l)
    A = np.matrix([[1,0,0,0],[-s2/((s1**2)+(s2**2)),s1/((s1**2)+(s2**2)),0,s4/(l*7*math.sqrt(1-x**2))],[s1,s2,0,0],[0,math.cos(e3),math.cos(e1),math.cos(e2)]])
    motors = -t*A.I*K*np.matrix([[s1-rT],[theta-thetaT],[(s1**2)+(s2**2)-(4*l*l)],[0]])
    print motors.item(0),motors.item(1),motors.item(2),motors.item(3)
    m1 = motors.item(2)*180/math.pi
    m2 = motors.item(1)*180/math.pi
    m3 = motors.item(0)*180/math.pi
    m4 = motors.item(3)*180/math.pi
    
    #print s1_t1-s1,s2_t1-s2,t
    m3 = m3/2.2869751969
    m2 = m2/2.2869751969
    m1 = m1/2.1964567
    m4 = m4/2.1964567
        
    print m1,m2,m3,m4
    dxl_io.set_moving_speed(dict(zip([1], itertools.repeat(512))))
    dxl_io.set_moving_speed(dict(zip([2], itertools.repeat(512))))
    dxl_io.set_moving_speed(dict(zip([3], itertools.repeat(512))))
    dxl_io.set_moving_speed(dict(zip([4], itertools.repeat(512))))
    a2=dxl_io.get_present_position(dict(zip([2], itertools.repeat(500))))
    a3=dxl_io.get_present_position(dict(zip([3], itertools.repeat(500))))
    a1=dxl_io.get_present_position(dict(zip([1], itertools.repeat(500))))
    a4=dxl_io.get_present_position(dict(zip([4], itertools.repeat(500))))
    dxl_io.set_goal_position(dict(zip([2], itertools.repeat(a2[0]-m2))))
    dxl_io.set_goal_position(dict(zip([3], itertools.repeat(a3[0]+m3))))
    dxl_io.set_goal_position(dict(zip([1], itertools.repeat(a1[0]+m1))))
    dxl_io.set_goal_position(dict(zip([4], itertools.repeat(a4[0]-m4))))
    

    time.sleep(0.02)
    #print rT,sd
