function code_from_text = retrive(r,w,bb)
code_from_text = cell(r ,w);
cc = textscan(bb , '%s','delimiter',',');

txt1 = reshape(cc{1,1} , w, r);
txt2 = transpose(txt1);

for i = 1 :  r
    for j = 1 :w
        a = txt2{i,j};
        b = size(a);
        d = zeros(b(2),1);

        for k = 1 :1: b(2)
            d(k,1) = str2num(a(1,k)) ;
            
        end
        code_from_text{i,j} = d ;     
    end    
end
end
