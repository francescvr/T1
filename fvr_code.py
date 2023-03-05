import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# EXERCICI 1

T= 2.5                                
fm=8000                              
fx=600                               
A=4                                  
pi=np.pi                             
L = int(fm * T)                      
Tm=1/fm                              
t=Tm*np.arange(L)                    
x = A * np.cos(2 * pi * fx * t)      
sf.write('so_exemple1.wav', x, fm)  

Tx = 1 / fx
Ls = int(fm * 5 * Tx)

plt.figure(0)
plt.plot(t[0:Ls], x[0:Ls])
plt.xlabel('t(s)')
plt.ylabel('Amp')
plt.title('5 periodes de la sinusoide')
plt.show()

from numpy.fft import fft
N = 5000
X = fft(x[0 : Ls], N)

K = np.arange(N)

plt.figure(1)
plt.subplot(211)
plt.plot(K, abs(X))
plt.title(f'Transformada del senyal de Ls = {Ls} mostres amb DFT de N = {N} ')
plt.ylabel('|X[k]|')
plt.subplot(212)
plt.plot(K, np.unwrap(np.angle(X)))
plt.xlabel('Index k')
plt.ylabel('$\phi_x[k]$')
plt.show()


import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


##Segona prova

T= 2.5                                
fm=12000                             
fx=6000                    
A=4                                  
pi=np.pi                             
L = int(fm * T)                      
Tm=1/fm                              
t=Tm*np.arange(L)                    
x = A * np.cos(2 * pi * fx * t)      
sf.write('so_exemple2.wav', x, fm)  

Tx = 1 / fx
Ls = int(fm * 5 * Tx)

plt.figure(2)
plt.plot(t[0:Ls], x[0:Ls])
plt.xlabel('t(s)')
plt.ylabel('Amp')
plt.title('5 periodes de la sinusoide')
plt.show()

from numpy.fft import fft
N = 5000
X = fft(x[0 : Ls], N)

K = np.arange(N)

plt.figure(3)
plt.subplot(211)
plt.plot(K, abs(X))
plt.title(f'Transformada del senyal de Ls = {Ls} mostres amb DFT de N = {N} ')
plt.ylabel('|X[k]|')
plt.subplot(212)
plt.plot(K, np.unwrap(np.angle(X)))
plt.xlabel('Index k')
plt.ylabel('$\phi_x[k]$')
plt.show()

#-----------------------------------------------------------------------------------

#EXERCICI 2

x_r, fm = sf.read('so_exemple2.wav')

plt.figure(4)
plt.xlabel('Hz')
ms = plt.magnitude_spectrum(x_r, fm)
fx = ms[1][np.argmax(ms[0])]
print(f'Freqüència fonamental: {fx}Hz')
plt.show()

 

Tx = 1 / fx
Ls = int(fm * 5 * Tx)

plt.figure(5)
plt.plot(t[0:Ls], x_r[0:Ls])
plt.xlabel('t(s)')
plt.ylabel('Amp')
plt.title('5 periodes de la sinusoide')
plt.show()

from numpy.fft import fft
N = 5000
X = fft(x_r[0 : Ls], N)

K = np.arange(N)

plt.figure(6)
plt.subplot(211)
plt.plot(K, abs(X))
plt.title(f'Transformada del senyal de Ls = {Ls} mostres amb DFT de N = {N} ')
plt.ylabel('|X[k]|')
plt.subplot(212)
plt.plot(K, np.unwrap(np.angle(X)))
plt.xlabel('Index k')
plt.ylabel('$\phi_x[k]$')
plt.show()

#-----------------------------------------------------------------------------------

##EXERCICI 3

x, fm = sf.read('so_exemple1.wav')


plt.figure(7)
plt.xlabel('Hz')
ms = plt.magnitude_spectrum(x, fm)
fx = ms[1][np.argmax(ms[0])]
print(f'Freqüència fonamental: {fx}Hz')
plt.show()

T = 2.5
L = int(fm*T)
Tm = 1 / fm
t = Tm * np.arange(L)
Tx = 1 / fx
Ls = int(fm * 5 * Tx)


N = fm
X = fft(x[0 : Ls], N)

X_db = 20*(np.log10(np.abs(X) / max(np.abs(X))))


K = np.arange(N)
k_ = (K/N) * fm


plt.figure(8)
plt.plot(k_[0 : int(fm / 2)], X_db[0 : int(fm /2)])
plt.title(f'Transformada del senyal de Ls={Ls} mostres amb DFT de N={N}')
plt.ylabel('|X[k]| dB')
plt.show()


A = 10**int(max(X_db)/20)
print(f'Amplitud del senyal: {A}')

#-----------------------------------------------------------------------------------

##EXERCICI 4


x_, fm = sf.read('Dance_Hall.wav')
nMostres = len(x_)
print(f'Freqüència mostratge: {fm}Hz')
print(f'Número de mostres: {nMostres}')

t_ = 5
L_ = int(fm * t_)


t_2 = 5.025
L_2 = int(fm * t_2)

Tm = 1 / fm
t = Tm * np.arange(L_, L_2)

plt.figure(9)
plt.title(f'Àudio 25 ms')
plt.plot(t, x_[L_, L_2])
plt.show()

N = fm
X_ = fft(x_[L_:L_2], N)
k__ = ((np.arange(N))/N) * fm
X_db_ = 20*np.log10(np.abs(X_)/(max(np.abs(X_))))

plt.figure (10)
plt.subplot(211)
plt.plot(k__[0:int(fm/2)] , X_db_[0:int(fm/2)])
plt.title(f'Transformada del senyal de Ls={L_2-L_} mostres amb DFT de N={N}')
plt.ylabel('|X[k]| en dB')
plt.subplot(212)
plt.plot(k__ / 2, np.unwrap(np.angle(X_db)))
plt.xlabel('K')
plt.ylabel('$\phi_x[k]$')
plt.show()