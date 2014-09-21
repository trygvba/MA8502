close all;
clear all;
%Script solving f_t + f_x = eps * f_xx, with f(0)=0 and f(1)=1,
%Using both upwind scheme and central differances.

%NOTE: Central differences should be stable only when dx<2eps.



%Setting parameters and domain:
eps = 0.01;
N = ceil(0.5/eps)-1;

x = linspace(0,1,N+2);
dx = x(2) - x(1);
dt = min(0.25*dx^2, dx);
T = 1;
nt = ceil(T/dt);

%Setting initial condition:
deg = 2;
u0 = x(2:(end-1)).^4;
%Stationary solution:
u_stat = (exp(x/eps)-1)/(exp(1/eps)-1);

%Starting time integration:
U_upwind = u0;
U_central = u0;
figure
for i=1:nt
    plot(x,[0 U_upwind 1],'r')
    hold on
    plot(x, [0 U_central 1], 'g')
    plot(x,u_stat,'b')
    legend('Upwind','Central diff.', 'Stationary')
    title(num2str(i/nt))
    hold off
    drawnow
    U_upwind = U_upwind -dt/dx * diff([0 U_upwind]) + eps*dt/dx^2 * diff([0 U_upwind 1],2);
    U_central = U_central -dt/(2*dx) * ([U_central(2:end) 1] - [0 U_central(1:(end-1))]) + eps*dt/dx^2 * diff([0 U_central 1],2);
    
end




