from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, TrainingArguments, Trainer
import torch
from peft import PeftModel

def load_summarized_model(model_path = 'model\ViT-base-summarize-text'):
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    original_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    peft_model = PeftModel.from_pretrained(original_model,
                                        model_path,
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)
    return tokenizer, peft_model

def summarize_document(text):
    tokenizer, peft_model = load_summarized_model()
    prompt = f"""
    Summarize the following text:
    Text: 
    {text}

    Summary:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=1000, num_beans = 1))
    summarized_document = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    return summarized_document

text = '''
Suốt bao năm, để dòng tranh này không bị rơi vào quên lãng, mỗi ngày người ta đều thấy ông Đạt cặm cụi làm nên những bức tranh từ mũi dao, cán đục. Ông bảo, tranh sơn khắc ở nước ta 
ra đời sớm nhất và còn đẹp hơn cả tranh sơn khắc của Nhật. Quý giá như vậy nên ông chẳng thể để nghề mai một trong sự chông chênh của thời cuộc. Một trong những sản phẩm sơn khắc của
ông Đạt được trả 25 triệu. Theo ông Đạt, thời điểm năm 1945 đến 1995 là lúc tranh sơn khắc ở nước ta phát triển mạnh nhất. Thời điểm đó, các sản phẩm của Hạ Thái chiếm tới 70% hàng 
xuất khẩu, giải quyết được công ăn việc làm cho người dân trong làng và cả các địa phương khác, đem lại cuộc sống khấm khá cho nhiều hộ gia đình. Say mê hội họa từ nhỏ, nên chuyện 
ông Đạt đến với tranh sơn khắc như một mối duyên tiền định. Khi mới tiếp xúc với những bức tranh này, ông Đạt như bị lôi cuốn chẳng thể nào dứt ra được. Học hết cấp 3, tôi thi vào 
Đại học sư phạm nhưng sức khỏe không đảm bảo nên xin vào làm thợ vẽ trong xưởng của hợp tác xã. Năm 1979, tôi được hợp tác xã cử đi học thêm ở trường Mỹ Nghệ. Khi về lại xưởng, nhờ 
năng khiếu hội họa nên tôi được chuyển sang khâu đoạn khảm trai rồi sang tranh khắc. Tôi làm tranh khắc từ đó đến giờ ông Đạt chia sẻ. Theo lời ông Đạt, học sơn khắc khó bởi cách 
vẽ của dòng tranh này khác hẳn với sơn mài. Nếu như sơn mài người ta có thể vẽ bằng chổi hay bút lông, cũng có khi là chất liệu mềm rồi mới quét sơn lên vóc thì sơn khắc khâu đoạn 
lại làm khác hẳn. Sơn khắc là nghệ thuật của đồ họa, sự hoàn thiện của bức tranh phụ thuộc vào những nét chạm khắc và những mảng hình tinh tế, giàu cảm xúc. Cuối cùng mới là việc 
tô màu nhằm tạo sự khắc họa mạnh. Như một lẽ xoay vần tự nhiên, sự phát triển của làng nghề Hạ Thái dần chùng xuống. Làng nghề bước vào thời kỳ suy thoái, đặc biệt là trong giai 
đoạn khủng hoảng kinh tế Đông Âu từ 1984 đến 1990 đã làm hợp tác xã tan rã. Ông Đạt khi đó cũng như bao người thợ khác đều phải quay về làm ruộng. Ông Đạt giải thích, tranh sơn khắc
xuất phát từ gốc tranh sơn mài. Nếu như ở tranh sơn mài thông thường, để có một tấm vóc vẽ người ta phủ sơn ta, vải lên tấm gỗ và mài phẳng thì tranh sơn khắc độc đáo ở chỗ, phải 
sử dụng kỹ thuật thủ công để khắc lên tấm vóc sơn mài. Tranh sơn khắc từ phôi thai, phác thảo đến lúc hoàn thành có khi kéo dài cả năm trời. Chẳng hạn, riêng công khắc ở bức tranh
khổ nhỏ thường tôi làm cả ngày lẫn đêm thì mất 2 ngày, phối màu mất 3 ngày. Để người trẻ học được nghề cũng sẽ mất khoảng 6 tháng đến 1 năm - ông Trần Thành Đạt chia sẻ. Tranh 
sơn khắc đòi hỏi rất kỹ về phác thảo, bố cục, cũng như mảng màu sáng tối mà màu đen của vóc là chủ đạo. Dù trên diện tích bức tranh khổ lớn bao nhiêu nó vẫn rất cần kỹ càng và 
chính xác đến từng xen-ti-met. Nếu sai, bức tranh sẽ gần như bị hỏng, các đường nét phải khắc họa lại từ đầu. Kỳ công là vậy nên giá thành mỗi sản phẩm sơn khắc thường khá cao, 
trung bình từ 4 đến 25 triệu đồng/bức tranh. Giá thành cao lại yêu cầu khắt khe về mặt kỹ thuật, mỹ thuật nên theo Nghệ nhân Trần Thành Đạt, nhiều người trong làng đã từ bỏ, 
không làm dòng tranh này nữa. Tranh sơn khắc làm mất nhiều thời gian và công sức nhưng khó bán. Họ đều tập trung làm tranh sơn mài, với chất liệu ngoại nhập cho rẻ và ít tốn 
công sức. Hầu như cả làng đã quay lưng, bỏ rơi dòng tranh sơn khắc vào lãng quên ông Đạt buồn bã kể. Được biết, hiện xưởng sản xuất tranh của ông Đạt chủ yếu là các thành viên 
trong gia đình. Ông khoe, hai con trai và con gái đều tốt nghiệp Trường Đại học Mĩ thuật, con rể và các con dâu cũng là họa sĩ của trường. Tất cả các thành viên trong gia đình 
ông đều chung niềm say mê với sơn khắc. Đinh Luyện.
'''