# function plot_I(I,r,fignum,ms)
# % I: intensity Nx3 matrix
# % r: coordinates Nx3 matrix
# % fignum: figure number
# % ms: marker size
#
# if nargin < 2
#   fignum = 1
# end
#
# I = I/max(max(I));
#
# figure(fignum)
# clf
# cm = colormap(jet(128));
#
# hold on
# for j = 1:length(I)
#   col_ind = round(I(j)*128);
#   if col_ind == 0
#     col_ind = 1;
#   end;
#   plot3(r(j,1),r(j,2),r(j,3),'.','MarkerSize',ms,'Color',cm(col_ind,:))
# end
# hold off
# zlabel('z')
# ylabel('y')
# xlabel('x')
# view(45,15)
# axis equal