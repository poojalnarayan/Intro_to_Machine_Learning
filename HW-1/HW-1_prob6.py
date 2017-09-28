import numpy as np
import matplotlib.pyplot as plt

def exercise6(infile='humu-2.txt', outfile='random.txt'):
    """
    Function that takes a file and converts it to numpy array and outputs another random elements file
    :param infile:humu-2.txt
    :param outfile:random.txt
    :return: generates 3plots. Function doesnt continue until you close them
    """
    arr = np.loadtxt(infile)
    print(type(arr))

    print(arr.size)
    #print(arr.shape)

#unpacking using shape, n-rows, m-columns in the i/p file
    n,m= arr.shape
    print(n,m, " shape")

    print("amin=", np.amin(arr), "  amax=", np.amax(arr))

#Min and Max values of the file humu-2.txt
    arrMin= np.amin(arr)
    arrMax= np.amax(arr)
    print(arrMin, arrMax, " Min and Max values of the original file/array")
    #plt.plot(arr[:,0], arr[:,1])

#Scaling the input file array to lie b/w [0,1]. (Each array element - min) / max-min
    newArr= (arr- arrMin)/(arrMax-arrMin)
    print(newArr.shape, "amin=", np.amin(newArr), "  amax=", np.amax(newArr), " scaled array")
    plt.figure()
    plt.imshow(arr)
#To print gray scale image
    plt.imshow(arr,cmap='gray')
    plt.show()

#Print the current colormap
    print(plt.cm.cmapname)

#creating random array and plotting
    randomArr = np.random.random((n,m))
    plt.figure()
    plt.imshow(randomArr,cmap='gray')
    plt.show()

#Writing the random array to a file
    np.savetxt('random.txt', randomArr, delimiter=' ')

#Loading the random array to another variable and displaying
    newRandArr = np.loadtxt(outfile)
    #print(type(arr))
    plt.figure()
    plt.imshow(newRandArr,cmap='gray')
    plt.show()

#Invoking the function
exercise6()

#Help on function defined with parameters
help(exercise6)

def exercise9():
    """
    Fucntion for Probabilty of both 6 on 2dice using random varilables with a seed
    :return:prints probability for 10trials
    """
#seed here is 8
    np.random.seed(seed=8)
#Generate random numbers from 2 dice for 1000o trials. ranges from [0,5]
#diceThrows = np.random.randint(6, size=(1000,2))
    #print(diceThrows.shape)
    for i in range(1,11):
        count=0
        diceThrows = np.random.randint(6, size=(1000, 2))
        for x,y in diceThrows:
            if x == 5 and y == 5:   #double sixes
                count = count + 1

        print("Trial ", i, "= ", count/1000)

#Invoking the function
exercise9()

def exercise10():
    """
    Setting the value to seed and operations on them
    :return:
    """
#Setting the random seed to 5 and generating 2 arrays to do a series of operations on them
    np.random.seed(seed=5)
    a = np.random.rand(3,1)
    b = np.random.rand(3,1)
    print("a= ",a)
    print("b= ", b)
    print("a+b= ",a+b)
    print("Hadamard product= ", np.multiply(a , b))
    print("dot-product= ",np.dot(a.T, b))

#10c:Setting the random seed to 2 and generating a matrix to do a series of operations on them
    np.random.seed(seed=2)
    arrX = np.random.rand(3,3)
    X= np.matrix(arrX)
    print("X= ",X)
    print("aT X= ",np.dot(a.T, X))
    print("aT X b= ",np.dot(np.dot(a.T, X), b))
    print("X inverse= ", X.I)

#Invoking the function 10
exercise10()

def exercise11():
    """
    plotting sine wave from 0.0 to 10.0
    :return: The plot can be saved
    """
    x = np.arange(0, 10, 0.01);
    y = np.sin(x)

    plt.figure()
    plt.xlabel("x values")
    plt.ylabel("sin(x)")
    plt.title("Sine Function for x from 0.0 to 10.0")
    plt.plot(x, y)
    plt.show()

#Invoking the function 11
exercise11()