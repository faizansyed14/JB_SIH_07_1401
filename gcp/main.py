from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import pyttsx3

model = None
interpreter = None
input_index = None
output_index = None

class_names = ['Anthracnose mango',
 'Bacterial Canker mango',
 'Bacterial spot rot cauliflower',
 'Bacterial spot tomato',
 'Black Rot cauliflower',
 'Black mold tomato',
 'Die Back mango',
 'Gall Midge mango',
 'Gray spot tomato',
 'Healthy cauliflower',
 'Healthy mango',
 'Late blight tomato',
 'Sooty Mould mango',
 'bacterial_blight_cotton',
 'bacterial_leaf_blight_Rice',
 'brown_spot_Rice',
 'cordana_Banana',
 'curl_virus_cotton',
 'fussarium_wilt_cotton',
 'health tomato',
 'healthy_Banana',
 'healthy_Rice',
 'healthy_cotton',
 'leaf_blast_Rice',
 'leaf_scald_Rice',
 'narrow_brown_spot_Rice',
 'pestalotiopsis_Banana',
 'powdery mildew tomato',
 'sigatoka_Banana']

BUCKET_NAME = "hexagon_leaf"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "Models/leafdisease.h5",
            "/tmp/leafdisease.h5",
        )
        model = tf.keras.models.load_model("/tmp/leafdisease.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    predicted_class = predicted_class
    confidence = round(100 * (np.max(predictions[0])), 2)

    dict = {       
  'Anthracnose mango':"Copper fungicides sprays (రాగి స్ప్రేలు)",
 'Bacterial Canker mango':"streptocycline200ppm sprays (స్ట్రెప్టోసైక్లిన్ స్ప్రే చేస్తుంది)",
 'Bacterial spot rot cauliflower':"sulfur sprays or Copper fungicides (సల్ఫర్ స్ప్రే)" ,
 'Bacterial spot tomato':"copper sprays & pathogen-free seed transplants (రాగి స్ప్రేలు)",
 'Black Rot cauliflower':"streptocycline (విత్తనాలను వేడి నీటిలో నానబెట్టడం)",
 'Black mold tomato':"copper sprays (రాగి స్ప్రేలు)",
 'Die Back mango':"Phosphite(ఫాస్ఫైట్) ",
 'Gall Midge mango':"magnesium sprays (మెగ్నీషియం స్ప్రే)",
 'Gray spot tomato':"amino acid sprays (అమైనో యాసిడ్ స్ప్రేలు)",
 'Healthy cauliflower':"No Need of Fertilizer(ఎరువులు అవసరం లేదు)",
 'Healthy mango':"No Need of Fertilizer(ఎరువులు అవసరం లేదు)",
 'Late blight tomato':"Mancozeb sprays (మాంకోజెబ్ స్ప్రే)",
 'Sooty Mould mango':"carbaryl or phosphomidon (కార్బరిల్ లేదా ఫాస్ఫామిడాన్)",
 'bacterial_blight_cotton':"Copper-based fungicides(రాగి ఆధారిత శిలీంద్రనాశకాలు)",
 'bacterial_leaf_blight_Rice':"Copper-based fungicides(రాగి ఆధారిత శిలీంద్రనాశకాలు)",
 'brown_spot_Rice':"Si fertilizers(Si ఎరువులు)",
 'cordana_Banana':"Cordana musae(కోర్డానా మ్యూసే)",
 'curl_virus_cotton':"lime-sulfur fungicide(సున్నం-సల్ఫర్ శిలీంద్ర సంహారిణి)",
 'fussarium_wilt_cotton':"nitrate nitrogen fertilizer(నైట్రేట్ నైట్రోజన్ ఎరువులు)",
 'health tomato':"No Need of Fertilizer(ఎరువులు అవసరం లేదు)",
 'healthy_Banana':"No Need of Fertilizer(ఎరువులు అవసరం లేదు)",
 'healthy_Rice':"No Need of Fertilizer(ఎరువులు అవసరం లేదు)",
 'healthy_cotton':"No Need of Fertilizer(ఎరువులు అవసరం లేదు)",
 'leaf_blast_Rice':"Use a protectant fungicide(రక్షిత శిలీంద్ర సంహారిణిని ఉపయోగించండి) ",
 'leaf_scald_Rice':"spraying of benomyl, fentin acetate(బెనోమిల్, ఫెంటిన్ అసిటేట్ చల్లడం)",
 'narrow_brown_spot_Rice':"Remove weeds and weedy rice (కలుపు మొక్కలు మరియు కలుపు బియ్యం తొలగించండి)",
 'pestalotiopsis_Banana':"foliar sprays of prochloraz (ప్రోక్లోరాజ్ యొక్క ఆకుల స్ప్రేలు)",
 'powdery mildew tomato':"baking soda liquid sprays (బేకింగ్ సోడా ద్రవ స్ప్రేలు)",
 'sigatoka_Banana':"Timorex Gold(టిమోరెక్స్ గోల్డ్)"

    }

    dict2 = {       
  'Anthracnose mango':"Anthracnose disease, caused by Colletotrichum gloeosporioides Penz. A fungicide is used to cure it. To determine whether it’s anthracnose, take a look at the underside of infected leaves with a magnifying glass.",
 'Bacterial Canker mango':": Canker disease of mango is caused by Xanthomonas campestris pv. Mangiferaeindicae. Agriculture sprays can be used to cure the disease.",
 'Bacterial spot rot cauliflower':": A solani, brassicicola, can cause leaf spot on cauliflower. It is also known as early leaf blight. Some of its preventions are Cleaning weeds, avoiding the environment for pathogens to grow etc.",
 'Bacterial spot tomato':"Bacterial spot of tomato is caused by Xanthomonas vesicatoria.It can be removed by symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants.",
 'Black Rot cauliflower':"Black rot of cauliflower caused by Xanthomonas campestris pv. Campestris. Hot water seed treatment and field sprays can be used to prevent this disease.",
 'Black mold tomato':"It is caused by Alternaria alternata. One way of preventing this disease is to avoid the repeated wetting and drying of the surface of the soil by using subsurface irrigation.",
 'Die Back mango':": Dieback are caused by many fungi and a few bacteria that produce stem or root rots. High temperature during summer predisposes the trees to the disease.",
 'Gall Midge mango':"Mango gall midge is spread by wind currents. Mixed and intercropping farming reduces the pest population. insecticides are recommended for against mango gall midge.",
 'Gray spot tomato':"It is caused by three different fungi,  Stemphylium solani, Stemphylium floridanum, and Stemphylium botryosum. It often strikes in warm, wet weather. Organic and chemical fungicides can be used to prevent this disease.",
 'Healthy cauliflower':"water it as per required",
 'Healthy mango':"water it as per required",
 'Late blight tomato':"It is caused by the water Mold Phytophthora infectants. Obtain tomato plants from trusted seed suppliers or reputable local sources to prevent late blight. Copper will kill all of these organisms.",
 'Sooty Mould mango':"The disease is of common occurrence and affects many kinds of fruits and plantation crops. It one of the species of fungi that grow on honey dew.",
 'bacterial_blight_cotton':"It is caused by xanthomanas axonopodis. It can be prevented by applying growth regulators.",
 'bacterial_leaf_blight_Rice':"It is a deadly and destructive bacterial disease. It is caused by a gram-negative bacteria named Xanthomanas oryzae.It is prevented by using proper fertiliser and keeping the paddy clean.",
 'brown_spot_Rice':"Brown spot disease in rice is caused by the fungus Cochliobolus miyabeanus. Practicing good sanitation prevents this disease.",
 'cordana_Banana':"It is called Cordana leaf spot, it is one of the most important fungal diseases of banana which helps in growth.It is caused by fungus.",
 'curl_virus_cotton':"The very characteristic symptoms include leaf curling, darkened veins, vein swelling and enations that frequently develop into cup-shaped, leaf-like structures on the undersides of leaves.",
 'fussarium_wilt_cotton':": Leaves on infected plants turn yellow and fall. The plant wilts over several days and then dies. A characteristic symptom of fusarium wilt is the reddish-brown discolouration of the water conducting tissue of the stem and roots",
 'health tomato':"water it as per required",
 'healthy_Banana':"water it as per required",
 'healthy_Rice':"water it as per required",
 'healthy_cotton':"water it as per required",
 'leaf_blast_Rice':"Blast  symptoms  appear  on  leaves  as  elliptical  spots with  light-colored  centers  and  reddish  edges.  The  most  serious  damage from  rice  blast  occurs  when  the  disease  attacks  the  nodes  just  below the  head,  often  causing  the  stem  to  break.",
 'leaf_scald_Rice':"zonate lesions of alternating light tan and dark brown starting from leaf tips or edges. oblong lesions with light brown halos in mature leaves. translucent leaf tips and margins.",
 'narrow_brown_spot_Rice':"caused by the fungus Cercospora janseana, varies in severity from year to year and is more severe as rice plants approach maturity. Leaf spotting may become very severe on the more susceptible varieties, and the disease causes severe leaf necrosis.",
 'pestalotiopsis_Banana':"The fungus causes leaf spots, petiole/rachis blights and sometimes bud rot of palms. Unlike other leaf spot and blight diseases, Pestalotiopsis palmarun attacks all parts of the leaf from the base to the tip",
 'powdery mildew tomato':"High humidity is favorable for the development of powdery mildew, but free water is not necessary for spore germination and infection. Disease development can occur in the presence or absence of dew. Excess nitrogen fertilization and high density stands may also be favorable for disease development.",
 'sigatoka_Banana':"Black sigatoka (Mycosphaerella fijiensis) first causes small, light yellow spots or streaks on leaves of about one month old. The symptoms run parallel to the veins. Within a few days, the spots become a few centimetres in size and turn brown with light grey centres."
    }

   
    


    pesticides = dict[predicted_class]
    discription = dict2[predicted_class]
    disc = discription
    speech = pesticides

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 120)
    audio = engine.say(speech)
    engine.runAndWait()
    



    return { 'class': predicted_class,
        'confidence': float(confidence),
        'pesticides':pesticides,
        'disc':disc,
        'speech':audio}

