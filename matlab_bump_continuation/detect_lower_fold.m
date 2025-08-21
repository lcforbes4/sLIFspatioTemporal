function ind = detect_lower_fold(Egrid)
ind = [];
    
    % Loop through the elements of Egrid, excluding the first and last elements
    for i = 2:length(Egrid)-1
        if Egrid(i) < Egrid(i-1) && Egrid(i) < Egrid(i+1)
            ind = i; %[ind, i]; % Add index to the result if the condition is met
        end
    end
end