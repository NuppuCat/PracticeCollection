function ChangeView(X_viewer,Y_viewer,Z_viewer,screen,scene)
viewer=[];

for i = 1:length(X_viewer)

    viewer.Location = [X_viewer(i), Y_viewer(i),Z_viewer(i)]; % [X,Y,Z]
    viewer.Orientation =  rotx(0) * roty(0) * rotz(0); % Identity matrix => eye(3) 
    %Viewer's extrinsic parameters
    viewer.R = eye(3);
    viewer.T = viewer.Location';   
    
    for k=1:length(scene)
        XYZ_data = scene{k}.vertices;
        screen.uv(k) = {Project3DTo2D2(XYZ_data, screen.K, viewer.R, viewer.T) };
    
        %Draw projected scene - virtual display
        %subplot(1,3,i)
        subplot(1,3,3)
        plot(screen.uv{k}(1,:), screen.uv{k}(2,:), '.', 'MarkerSize', 20);
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
        for c = 1:size(scene{k}.connectivity, 2)
            patch(  'Faces', [1 2 3], ...
                'Vertices', [screen.uv{k}(:,scene{k}.connectivity(1, c)), ...
                screen.uv{k}(:,scene{k}.connectivity(2, c)), ...
                screen.uv{k}(:,scene{k}.connectivity(3, c))]', ...
                'FaceColor', scene{k}.color(:,c), ...
                'EdgeColor', 'none');
        end
        title("x= "+num2str(X_viewer(i))+" y="+num2str(Y_viewer(i))+" z="+num2str(Z_viewer(i)));
        axis image;
        xlim([0,1920]);
        ylim([0,1080]);
        
    end
    hold off;
    pause(0.1);
end
end

