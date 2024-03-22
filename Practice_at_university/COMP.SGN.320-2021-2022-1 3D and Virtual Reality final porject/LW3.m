%--------------------------------------------------------------------------
% COMP.SGN.320 3D and Virtual Reality
%
%
% Your implementation should run by executing this m-file ("run LW3.m"), 
% but feel free to create additional files for your own functions
% Make sure it runs without errors after unzipping
%
% Group members: harri.p.lehtonen@tuni.fi, martin.moritz@tuni.fi,
% yingqi.zhao@tuni.fi
% Additional tasks completed (2.4, 2.5, 2.6, 2.7, 2.8):
% Completed 2.5, 2.6, 2.7. Task 2.5 is plotted along with 2.2, 2.6
% stabilization is shown along with 2.3 and z-buffer is its own section.
%
% SUGGEST RUNNING THE SCRIPT ONE SECTION AT A TIME!
%
% The tracking system and image rendering should run in real-time (> 1 fps)
%--------------------------------------------------------------------------
%% Model creation - Task 2.1
clear all;
close all;
% Define model (cube):

cube = [];

cube.vertices = [-1,1,1,-1,-1,1,-1,1;-1,-1,1,1,1,1,-1,-1;1,1,1,1,-1,-1,-1,-1];    %X,Y,Z

cube.connectivity = [1,1,4,4,4,4,5,5,6,6,5,5;2,7,3,5,3,1,4,7,3,8,6,7;8,8,6,6,2,2,1,1,2,2,8,8];


cube.color = [rand(1,12);rand(1,12);rand(1,12)]

%Transform the model:          
model1 = cube;
%Scale factor
scale = 0.05;
%Rotation matrix
rotation = rotX(15) * rotY(30) * rotZ(20);
%Translation vector
translation = [0.15,0.15,0.1];
%Apply transformations
model1.vertices = model1.vertices .* scale;
model1.vertices = rotation * model1.vertices;
model1.vertices = model1.vertices + translation';

%Transform the model:          
model2 = cube;
%Scale factor
scale2 = 0.04;
%Rotation matrix
rotation2 = rotX(80) * rotY(30) * rotZ(90);
%Translation vector
translation2 = [0.1,0.13,0.3];
%Apply transformations
model2.vertices = model2.vertices .* scale2;
model2.vertices = rotation2 * model2.vertices;
model2.vertices = model2.vertices + translation2';
%Transform the model:          
model3 = cube;
%Scale factor
scale3 = 0.03;
%Rotation matrix
rotation3 = rotX(40) * rotY(3) * rotZ(63);
%Translation vector
translation3 = [-0.1,0.2,0.2];
%Apply transformations
model3.vertices = model3.vertices .* scale3;
model3.vertices = rotation3 * model3.vertices;
model3.vertices = model3.vertices + translation3';

%The scene
scene = {model1, model2, model3};

%Draw scene, triangle-by-triangle

for(k=1:length(scene))
    for c = 1:size(scene{k}.connectivity, 2)
        patch('Faces',[1 2 3], ...
            'Vertices',[scene{k}.vertices(:,scene{k}.connectivity(1, c)), ...
            scene{k}.vertices(:,scene{k}.connectivity(2, c)), ...
            scene{k}.vertices(:,scene{k}.connectivity(3, c))]', ...
            'FaceColor', scene{k}.color(:,c), ...
            'EdgeColor', 'k');
        
        hold on;
        
    end
end

%Draw axes - sensor origin
DrawAxes(eye(3), [0,0,0], 1 ,'Sensor Origin', 0.01, []);

%Set title, axes format, limits, etc
title('Model creation - Task 2.1');
axis image;
grid on;
view(45,36);
xlim([-1.2,1.2]); ylim([-1.2,1.2]); zlim([-1.2,1.2]);  
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');

%% Perspective projection - Task 2.2 and 2.5

%Screen properties:
screen = [];
%Resolution (native resolution)
resx = 1920;
resy = 1080;
pix_size = 0.00028;
physx = 0.543;
physy = 0.303;
screen.res = [resx, resy]; % [u,v]
%Screen's pixel size - force pixel to be square!  
screen.pixelSize = [ pix_size, pix_size]; % [x,y] 
%Screen physical size (in meters)
screen.physicalSize = [ physx, physy]; % [x,y]
%Screen 3D coordinates
screen.coord3D = [physx/2 physx/2 -physx/2 -physx/2; ... %X 
                  0 physy physy 0; ... %Y
                  0 0 0 0];    %Z

%Screen f (distance from viewer to screen)
screen.f = [ 0.5/pix_size, 0.5/pix_size]; % [fx,fy]
%Screen principal point
screen.pp = [ resx/2, resy/2]; %[cx, cy]
%Screen intrinsic parameters matrix
screen.K = [screen.f(1), 0, screen.pp(1); ...
           0, screen.f(2), screen.pp(2); ...
           0, 0, 1];
%Projected 3D points for each model in the scene       
screen.uv = cell([1, length(scene)]);

%Viewer properties:       
viewer = [];
%Viewer's pose
viewer.Location = [0, 0.1512 , -0.5]; % [X,Y,Z]
viewer.Orientation = rotX(0) * rotY(0) * rotZ(0); % Identity matrix => eye(3) 
%Viewer's extrinsic parameters
viewer.R = eye(3);
viewer.T = [0; -0.1512; 0.5];   

array_for_ordering = [];
for(k=1:length(scene))
    
    %Project
    XYZ_data = scene{k}.vertices;
    screen.uv(k) = {Project3DTo2D(XYZ_data, screen.K, viewer.R, viewer.T) };
    
    
    %Draw scene
    subplot(2,2,[1,3]);
    for c = 1:size(scene{k}.connectivity, 2)
        patch('Faces',[1 2 3], ...
            'Vertices',[scene{k}.vertices(:,scene{k}.connectivity(1, c)), ...
            scene{k}.vertices(:,scene{k}.connectivity(2, c)), ...
            scene{k}.vertices(:,scene{k}.connectivity(3, c))]', ...
            'FaceColor', scene{k}.color(:,c), ...
            'EdgeColor', 'none');
        
        hold on;
    end
    
    %Draw 3D screen
    line([screen.coord3D(1,:), screen.coord3D(1,1)], ...
         [screen.coord3D(2,:), screen.coord3D(2,1)], ...
         [screen.coord3D(3,:), screen.coord3D(3,1)], ...
         'Color',[0,0,0.3], 'LineWidth', 2);
    
    %Draw axes - sensor origin
    DrawAxes(eye(3), [0,0,0], 1 ,'Sensor Origin', 0.01);
    
    %Draw viewer
    DrawAxes(viewer.Orientation, viewer.Location, 0.2 ,'Viewer', 0.01);
    
    %Set title, axes format, limits, etc
    title('Scene');
    axis image;
    grid on;
    view(33,22);
    xlim([-1.2,1.2]); ylim([-1.2,1.2]); zlim([-1.2,1.2]);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    
    
    %Draw projected scene - virtual display
    subplot(2,2,2)
    plot(screen.uv{k}(1,:), screen.uv{k}(2,:), '.', 'MarkerSize', 10);
    set(gca, 'YDir', 'reverse');
    hold on;
    

    zetat = scene{k}.vertices(3,:);
    xt = scene{k}.vertices(1,:);
    yt = scene{k}.vertices(2,:);
    for n = 1:12
        z1 = zetat(scene{k}.connectivity(1,n));
        z2 = zetat(scene{k}.connectivity(2,n));
        z3 = zetat(scene{k}.connectivity(3,n));
        z_avg = (z1+z2+z3)/3;
        x1 = xt(scene{k}.connectivity(1,n));
        x2 = xt(scene{k}.connectivity(2,n));
        x3 = xt(scene{k}.connectivity(3,n));
        x_avg = (x1+x2+x3)/3;
        y1 = yt(scene{k}.connectivity(1,n));
        y2 = yt(scene{k}.connectivity(2,n));
        y3 = yt(scene{k}.connectivity(3,n));
        y_avg = (y1+y2+y3)/3; 
        euc_dist = sqrt((viewer.Location(1)-x_avg)^2+(viewer.Location(2)-y_avg)^2+(viewer.Location(3)-z_avg)^2);
        scene{k}.connectivity(4,n) = euc_dist;
    end
    
    [temp, order] = sort(scene{k}.connectivity(4,:),'descend');
    scene{k}.connectivity = scene{k}.connectivity(:,order);
    scene{k}.color = scene{k}.color(:,order);

    array_for_ordering = [array_for_ordering, scene{k}.connectivity(4,1)]; %Used for 2.5
    for c = 1:size(scene{k}.connectivity, 2)
        patch(  'Faces', [1 2 3], ...
            'Vertices', [screen.uv{k}(:,scene{k}.connectivity(1, c)), ...
            screen.uv{k}(:,scene{k}.connectivity(2, c)), ...
            screen.uv{k}(:,scene{k}.connectivity(3, c))]', ...
            'FaceColor', scene{k}.color(:,c), ...
            'EdgeColor', 'none');
    end
    title('Perspective projection - Task 2.2');
    axis image;
    xlim([0,1920]);
    ylim([0,1080]);
    
end
% Code for 2.5
[temp, polygon_order] = sort(array_for_ordering,'descend');
for k = polygon_order
    
    subplot(2,2,4)
    plot(screen.uv{k}(1,:), screen.uv{k}(2,:), '.', 'MarkerSize', 10);
    set(gca, 'YDir', 'reverse');
    hold on;
    for c = 1:size(scene{k}.connectivity, 2)
        patch(  'Faces', [1 2 3], ...
            'Vertices', [screen.uv{k}(:,scene{k}.connectivity(1, c)), ...
            screen.uv{k}(:,scene{k}.connectivity(2, c)), ...
            screen.uv{k}(:,scene{k}.connectivity(3, c))]', ...
            'FaceColor', scene{k}.color(:,c), ...
            'EdgeColor', 'none');
    end
    title('Perspective projection - Task 2.5');
    axis image;
    xlim([0,1920]);
    ylim([0,1080]);
end
%% Changing viewpoint - Task 2.3 and 2.6
%The first plot of cubes is without jitter stabilization, the second one shows both
%coordinates and plots stabilized cubes.

X_viewer=[-0.5:0.01:0.5];
% do y as random, so can check the perform of jitter stabilization
Y_viewer=[-0.5:0.01:0.5]+[rand(1,101)*0.1];
Z_viewer=[-0.5:0.004:-0.1];
viewbuffer = zeros([3,4]);
figure;
%No stabilization
for i = 1:length(X_viewer)
   
    x1 = X_viewer(i);y1=Y_viewer(i);z1=Z_viewer(i);
    subplot(1,3,1)
    plot(i,x1,'r--o',i,y1,'g--x',i,z1,'b--*');hold on;
    title('Perspective projection - Task 2.3 without jitter stabilization');
    ChangeView(X_viewer(i),Y_viewer(i),Z_viewer(i),screen,scene);
        
end
figure;
for i = 1:length(X_viewer)
    viewbuffer(:,1)=viewbuffer(:,2);
    viewbuffer(:,2)=viewbuffer(:,3);
    viewbuffer(:,3)=viewbuffer(:,4);
    viewbuffer(:,4)= [X_viewer(i) Y_viewer(i) Z_viewer(i)]';
    vt = EWA(viewbuffer);
    vt = vt(:);
    x1 = X_viewer(i);y1=Y_viewer(i);z1=Z_viewer(i);
    subplot(1,3,1)
    plot(i,x1,'r--o',i,y1,'g--x',i,z1,'b--*');hold on;
    subplot(1,3,2)
    plot(i,vt(1),'r--o',i,vt(2),'g--x',i,vt(3),'b--*');hold on;drawnow;
    title('Perspective projection - Task 2.3 with jitter stabilization(2.5)');
    ChangeView(vt(1),vt(2),vt(3),screen,scene);
        
end

%% Z-buffer rendering - Task 2.7

frame = 3; %Viewangle can be changed within [1:101]
viewer=[];
viewer.Location = [X_viewer(frame), Y_viewer(frame),Z_viewer(frame)]; % [X,Y,Z]
viewer.Orientation =  rotx(0) * roty(0) * rotz(0); % Identity matrix => eye(3) 
%Viewer's extrinsic parameters
viewer.R = eye(3);
viewer.T = viewer.Location'; 
for k=1:length(scene)
     XYZ_data = scene{k}.vertices;
     screen.uv(k) = {Project3DTo2D2(XYZ_data, screen.K, viewer.R, viewer.T) };
end
[raster_image,depthmap] = zbuffer(scene,screen);
%Plotting
for(k=1:length(scene))

    %Draw scene
    subplot(2,2,[1,3]);
    for c = 1:size(scene{k}.connectivity, 2)
        patch('Faces',[1 2 3], ...
            'Vertices',[scene{k}.vertices(:,scene{k}.connectivity(1, c)), ...
            scene{k}.vertices(:,scene{k}.connectivity(2, c)), ...
            scene{k}.vertices(:,scene{k}.connectivity(3, c))]', ...
            'FaceColor', scene{k}.color(:,c), ...
            'EdgeColor', 'none');
        
        hold on;
    end
    
    %Draw 3D screen
    line([screen.coord3D(1,:), screen.coord3D(1,1)], ...
         [screen.coord3D(2,:), screen.coord3D(2,1)], ...
         [screen.coord3D(3,:), screen.coord3D(3,1)], ...
         'Color',[0,0,0.3], 'LineWidth', 2);
    
    %Draw axes - sensor origin
    DrawAxes(eye(3), [0,0,0], 1 ,'Sensor Origin', 0.01);
    
    %Draw viewer
    DrawAxes(viewer.Orientation, viewer.Location, 0.2 ,'Viewer', 0.01);
    
    %Set title, axes format, limits, etc
    title('Scene');
    axis image;
    grid on;
    view(33,22);
    xlim([-1.2,1.2]); ylim([-1.2,1.2]); zlim([-1.2,1.2]);
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    
end
subplot(2,2,2)
imshow(depthmap,[1 800])
title("Z.buffer rendering (frame "+string(frame)+")")
subplot(2,2,4)
imshow(raster_image)
sgtitle('Task 2.7 (viewangle can be changed by changing paramater frame within [1:101])')
%%
%--------------------------------------------------------------------------
%Functions
%--------------------------------------------------------------------------
function  Rx = rotX(angle) %in degrees
Rx = [1 0 0 ; ...
      0 cosd(angle) -sind(angle); ...
      0 sind(angle) cosd(angle)];
end

function  Ry = rotY(angle) %in degrees
Ry = [cosd(angle) 0 sind(angle); ...
      0 1 0; ...
      -sind(angle) 0 cosd(angle)];
end

function  Rz = rotZ(angle) %in degrees
Rz = [cosd(angle)  -sind(angle) 0; ...
     sind(angle) cosd(angle) 0; ...
      0 0 1];
end