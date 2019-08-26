function show_images(X,Y,N_l,conf_thresh,gaussian_filtering,sigma)
% this function plots sample labeled images from each class:
% inputs:
% x: images
% Y: labels matrix
% N_l: number of classes
% conf_thresh: nofidence threshold for the labeling
% NOTE: THIS FUNCTION ASSUMES EACH CLASS HAS SAME # OF IMAGES!


imgPerClass = 6; % how many images to show for each class
indexes = round(linspace(1,size(X,2), N_l*imgPerClass));

idx = 0; % img index
close all;
figure;
hold on;

for i = 1:N_l
    for j = 1:imgPerClass
        idx = idx+1;
        img_index = indexes(idx);
        
        img = reshape(squeeze(X(:,img_index)),[99 99]);
        Y_idx = squeeze(Y(img_index,:,:));
        
        if gaussian_filtering == 1
            for k = 1:N_l
                temp = reshape(squeeze(Y_idx(:,k)),[99,99]);
                temp = imgaussfilt(temp,sigma);
                Y_idx(:,k) = reshape(temp,[99*99,1]);
            end
        end
        [vals, classifiedImage] = max(Y_idx,[],2);
        % vals here can be useful for showing which areas have more
        % confidence values
        conf = vals./(sum(Y_idx,2)+1e-6);
        classifiedImage =  reshape(classifiedImage,99,99);
        conf =  reshape(conf,99,99);
                
        % comment out this line to remove the median filtering of the results: 
        classifiedImage = medfilt2(classifiedImage,[3,3],'symmetric');
        classifiedImage(conf<conf_thresh) = 0;
        
        coloredImage =  uint8(zeros([size(img),3]));
        for ii = 1:size(img,1)
            for jj = 1:size(img,2)
                if classifiedImage(ii,jj) == 0     % Not Sure:
                    coloredImage(ii,jj,:) = [183,183,183]; % Gray
                elseif classifiedImage(ii,jj) == 1     % Chaotic:
                    coloredImage(ii,jj,:) = [0,0,255]; % blue
                elseif classifiedImage(ii,jj) == 2  % Other:
                    coloredImage(ii,jj,:) = [100,255,255]; % light blue
                elseif classifiedImage(ii,jj) == 3 % Fault:
                    coloredImage(ii,jj,:) = [0,255,0]; % Green
                elseif classifiedImage(ii,jj) == 4 % Salt:
                    coloredImage(ii,jj,:) = [255,0,0]; % red
                end
            end
        end
        
        clear classifiedImage
        
        subplot(N_l,imgPerClass,idx)
        imshow(img,[]);
        hold on;
        g = imagesc(coloredImage);
        set(g, 'AlphaData', 0.45 );
        
    end
end

end

