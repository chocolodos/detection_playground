import socket   #for sockets
import sys  #for exit

def convert(args):
    #args = [ -0.5404195,   0.52105546,  0.06703898,  0.  ,        0.98270189 , 2.60117315]

    degs = []
    for arg in args:
        deg = arg*180/3.1415926
        degs.append(deg)

    strResult1 = "[{0:.3f},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f}]".format(degs[0],degs[1],degs[2],degs[3],degs[4],degs[5])
    #print strResult1

    return strResult1

def SendMessage(s,message):
    try :
        #Set the whole string
        s.sendall(message)
    except socket.error:
        #Send failed
        print ('Send failed')
        sys.exit()

    StrResult = s.recv(1024)
    print ('Message = ' + message + ' Sending Successfully')

    return StrResult

def Setup():
    try:
        #create an AF_INET, STREAM socket (TCP)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except (socket.error, msg):
        print ('Failed to create socket. Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1])
        sys.exit()

    print ('Socket Created')

    host = '10.137.209.237'
    #host = '127.0.0.1'
    port = 8791

    try:
        remote_ip = socket.gethostbyname( host )
    except socket.gaierror:
        #could not resolve
        print ('Hostname could not be resolved. Exiting')
        sys.exit()

    print ('Ip address of host' + host + ' is ' + remote_ip)

    #Connect to remote server
    s.connect((remote_ip , port))

    print ('Socket Connected to ' + host + ' on ip ' + remote_ip)
    return s


def StartPath(s):
    #Send some data to remote server
    message = "pathstart"
    strBack = SendMessage(s,message)

def StartPoints(s,points):
    for point in points:
        strPoint = convert(point)
        #print strPoint
        SendMessage(s,strPoint)

def ClosePath(s):

    message = "pathend"
    strBack = SendMessage(s,message)

def CloseSocket(s):
    s.close()
