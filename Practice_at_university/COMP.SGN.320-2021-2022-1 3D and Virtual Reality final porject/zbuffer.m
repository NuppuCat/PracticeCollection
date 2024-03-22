function [raster_image,depthmap] = zbuffer(scene,screen)
    % calculate the raster cloured image and the depth map given a 
    % scene and a screen
    
    width = screen.res(1);
    height = screen.res(2);
    
    depthmap = ones(width*height,1)*100000;
    raster_image = ones(width*height,3);
    
    
    for s = 1:3
        XYZ = scene{s}.vertices;
        triangles = scene{s}.connectivity;
        colours = scene{s}.color;
        UV = screen.uv{s};
        
        for n_triangle = 1:size(triangles,2)
            % a depthmask and a coloured mask are computed for each triangle
            depthmask = ones(width*height,1)*100000;
            colourmask = zeros(width*height,3);
        
        
            triangle = UV(:,triangles(1:3,n_triangle));
            % the pixels to consider are the one inside the triangle and
            % inside the screen
            box = int16(minmax(triangle));
            box(1,1) = max(box(1,1),1);
            box(2,1) = max(box(2,1),1);
            box(1,2) = min(box(1,2),width);
            box(2,2) = min(box(2,2),height);
            
            % creates the list of the pixels inside the box
            [u,v] = meshgrid([box(1,1):box(1,2)],[box(2,1):box(2,2)]);            
            pixels = [reshape(u,[],1) , reshape(v,[],1)]' ;
            pixels = double(reshape(pixels,2,1,[]));    

            % for each pixel, calculate the centroid coordinates (coeff)
            % related to the vertices of the triangle
            diff = pixels - triangle;
            coeff = ones(3,size(pixels,3));
            for i=1:size(pixels,3)
                coeff(2:3,i) = diff(:,2:3,i) \ -diff(:,1,i);
            end
            coeff = coeff ./ sum(coeff,1);
            
            pix = reshape(pixels,2,[]);
            % a pixel is inside the triangle if all the three centroid coordinates are positive. 
            insiders =coeff(1,:)>0 & coeff(3,:)>0 & coeff(2,:)>0;
            
            % calculate the depth in millimeters of the vertices of the
            % triangle
            depthTriangle = XYZ(3,triangles(1:3,n_triangle))*1000;
            
            pix = int16(pix(:,insiders));

            % get the linear indices of the good pixels (the ones inside
            % the triangle)
            pix_ind = sub2ind([height,width],pix(2,:),pix(1,:));
            
            % calculate the depth for each pixel, based on the centroid
            % coefficients and on the depth of the vertices
            depths = depthTriangle * coeff;
            % create the mask for this triangle
            depthmask(pix_ind) = depths(insiders);
            colourmask(pix_ind,:)  = repmat( colours(:,n_triangle)' , size(pix_ind,2),1);
            % update the ratser coloured image and the depthmap
            raster_image(depthmask < depthmap,:) = colourmask(depthmask < depthmap,:);
            depthmap(depthmask < depthmap) = depthmask(depthmask < depthmap);
        
        end
    end

    % puts the raster image and the depthmap to the correct image size
    raster_image = reshape(raster_image,[],1,3);
    raster_image = reshape(raster_image,height,width,3);
    
    depthmap = reshape(depthmap,height,width);
end

