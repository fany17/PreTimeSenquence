while 1
    try
        C_SVM_Real;
        send_wechat_message(string(datetime('now'))+" : "+positionpred)
        send_temp_image_to_wechat()
    catch ME
        disp('Error');
    end
    pause(10);
end


function send_wechat_message(message)
    % 企业微信机器人 Webhook 地址从环境变量读取，避免提交明文密钥
    webhook = getenv('WECHAT_WEBHOOK_URL');
    if strlength(webhook) == 0
        error('WECHAT_WEBHOOK_URL is not set.');
    end
    
    % 创建消息内容
    json_data = jsonencode(struct('msgtype', 'text', ...
                                  'text', struct('content', message)));
    
    % 设置 HTTP 选项
    options = weboptions('MediaType', 'application/json', 'RequestMethod', 'post');
    
    % 发送 HTTP 请求
    response = webwrite(webhook, json_data, options);
    disp('Message sent successfully!');
    disp(response);
end


function send_temp_image_to_wechat()
    % 企业微信机器人 Webhook 地址从环境变量读取，避免提交明文密钥
    webhook = getenv('WECHAT_WEBHOOK_URL');
    if strlength(webhook) == 0
        error('WECHAT_WEBHOOK_URL is not set.');
    end

    % 图片路径
    image_path = 'temp.png';
    
    % 读取图片并进行 Base64 编码
    fileID = fopen(image_path, 'rb');
    image_data = fread(fileID, '*uint8')';
    fclose(fileID);
    image_base64 = matlab.net.base64encode(image_data);
    
    % 计算 MD5 值
    md5_hash = java.security.MessageDigest.getInstance('MD5');
    md5_hash.update(image_data);
    image_md5 = sprintf('%02x', typecast(md5_hash.digest(), 'uint8'));
    
    % 构造 JSON 数据
    json_data = jsonencode(struct(...
        'msgtype', 'image', ...
        'image', struct(...
            'base64', image_base64, ...
            'md5', image_md5 ...
        ) ...
    ));
    
    % 设置 HTTP 请求选项
    options = weboptions('MediaType', 'application/json', 'RequestMethod', 'POST');
    
    % 发送 HTTP POST 请求
    response = webwrite(webhook, json_data, options);
    
    % 显示发送结果
    disp('Image sent successfully!');
    disp(response);

end





