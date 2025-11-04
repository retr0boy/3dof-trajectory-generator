% Link lengths
L1 = 135;
L2 = 135;
L3 = 46.7;

% Define joint angles (in degrees)
theta1 = -45:1:45;  % use step 10 to keep loops reasonable
theta2 = -45:1:45;
theta3 = -45:1:45;

% Preallocate arrays
x = zeros(length(theta1), length(theta2), length(theta3));
y = zeros(length(theta1), length(theta2), length(theta3));

% Nested loops for each joint
for i = 1:length(theta1)
    for j = 1:length(theta2)
        for k = 1:length(theta3)
            x(i,j,k) = L1*cosd(theta1(i)) ...
                     + L2*cosd(theta1(i)+theta2(j)) ...
                     + L3*cosd(theta1(i)+theta2(j)+theta3(k));

            y(i,j,k) = L1*sind(theta1(i)) ...
                     + L2*sind(theta1(i)+theta2(j)) ...
                     + L3*sind(theta1(i)+theta2(j)+theta3(k));
        end
    end
end

% Flatten and plot
plot(x(:), y(:), 'b.');
axis equal;
grid on;
xlabel('X position');
ylabel('Y position');
title('Reachable workspace of 3-link manipulator');

