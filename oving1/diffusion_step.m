function U_new = diffusion_step(U_old)
    U_new = diff([0 U_old 1],2);
end