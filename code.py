import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y, yn
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

def Y(l, m, theta, phi):
    return sph_harm_y(l, m, theta, phi)

theta = np.linspace(0, np.pi, 300)
phi   = np.linspace(0, 2*np.pi, 600)
THETA, PHI = np.meshgrid(theta, phi)

fig1, ax1 = plt.subplots(figsize=(8, 5))
x = np.linspace(0.01, 20, 2000)
for n, color, label in zip([0,1,2], ['#2196F3','#E91E63','#4CAF50'],
                            [r'$N_0(x)$', r'$N_1(x)$', r'$N_2(x)$']):
    yy = np.clip(yn(n, x), -3, 3)
    ax1.plot(x, yy, color=color, lw=2, label=label)
ax1.axhline(0, color='black', lw=0.7, ls='--')
ax1.set_xlim(0.01, 20); ax1.set_ylim(-3, 1.5)
ax1.set_xlabel(r'$x$', fontsize=13); ax1.set_ylabel(r'$N_n(x)$', fontsize=13)
ax1.set_title('Funciones de Neumann', fontsize=13)
ax1.legend(fontsize=12); ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('neumann_functions.png', dpi=200, bbox_inches='tight')
plt.close()

modes = [(1,0),(2,0),(2,1),(4,0),(4,2),(4,4),(10,0),(10,5),(10,10)]
fig2, axes = plt.subplots(3, 3, figsize=(13, 9))
fig2.suptitle('Armonicos esfericos - parte real', fontsize=14)
for ax, (l, m) in zip(axes.flat, modes):
    Z = np.real(Y(l, m, THETA, PHI))
    im = ax.imshow(Z.T, extent=[0,360,0,180], cmap='RdBu_r', aspect='auto',
                   vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)))
    ax.set_title(f'l={l}, m={m}', fontsize=10)
    ax.set_xlabel('phi', fontsize=8); ax.set_ylabel('theta', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig('spherical_harmonics_2d.png', dpi=200, bbox_inches='tight')
plt.close()

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Multipolos bajos vs altos', fontsize=13)
np.random.seed(42)
for ax, lmax, title in zip(axes3, [5, 30],
    ['Multipolos bajos (l<=5)\nEstructuras grandes',
     'Multipolos altos (l<=30)\nEstructuras finas']):
    T_map = np.zeros((300, 600))
    for l in range(1, lmax+1):
        for m in range(-l, l+1):
            alm = (np.random.randn() + 1j*np.random.randn()) / (l*(l+1))**0.5
            T_map += np.real(alm * Y(l, m, THETA, PHI)).T
    im = ax.imshow(T_map, extent=[0,360,0,180], cmap='RdBu_r', aspect='auto')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('phi', fontsize=9); ax.set_ylabel('theta', fontsize=9)
    plt.colorbar(im, ax=ax, label='Delta T (u.a.)', fraction=0.03)
plt.tight_layout()
plt.savefig('cmb_multipoles_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

fig4, axes4 = plt.subplots(1, 3, figsize=(14,5), subplot_kw={'projection':'3d'})
fig4.suptitle('Armonicos esfericos 3D', fontsize=13)
for ax, (l,m) in zip(axes4, [(2,0),(4,2),(6,3)]):
    Yval = np.real(Y(l, m, THETA, PHI))
    R = np.abs(Yval)
    X3 = R * np.sin(THETA) * np.cos(PHI)
    Y3 = R * np.sin(THETA) * np.sin(PHI)
    Z3 = R * np.cos(THETA)
    C = (Yval - Yval.min())/(Yval.max()-Yval.min())
    ax.plot_surface(X3.T, Y3.T, Z3.T, facecolors=plt.cm.RdBu_r(C.T),
                    alpha=0.9, rstride=4, cstride=4)
    ax.set_title(f'l={l}, m={m}', fontsize=12)
    ax.set_axis_off()
plt.tight_layout()
plt.savefig('spherical_harmonics_3d.png', dpi=200, bbox_inches='tight')
plt.close()