function U_new = upwind_step(U_old)
    U_new = diff([0 U_old]);
end