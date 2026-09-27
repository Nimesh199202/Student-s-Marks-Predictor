import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title='Student Marks Predictor', page_icon='📚', layout='wide')

st.markdown('''<style>
.main-title{text-align:center;font-size:38px;font-weight:700;margin-bottom:5px}.subtitle{text-align:center;font-size:17px;margin-bottom:25px}.result-box{padding:20px;border-radius:12px;text-align:center;border:1px solid rgba(128,128,128,.35);margin-top:15px}.result-number{font-size:42px;font-weight:700}div[data-testid="stTabs"] button{font-size:18px;font-weight:600}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None, f'File not found: {path}'
    try: return joblib.load(path), None
    except Exception as e: return None, str(e)

math_model, math_error = load_model('gradient_boost_mathes_finalized.pkl')
science_model, science_error = load_model('gbr_science_model.pkl')
science_scaler, scaler_error = load_model('gbr_science_scaler.pkl')

family_income_mapping={1:'Below 25,000',2:'25,000 - 50,000',3:'50,000 - 75,000',4:'75,000 - 100,000',5:'Above 100,000'}
edu_mapping={0:"Don't go to school",1:'Grade (1-5)',2:'Grade (6-11)',3:'Pass (O/L)',4:'Diploma',5:'Pass (A/L)',6:'Degree holder'}
relationship_mapping={0:'Very bad / Bad',1:'Average',2:'Good',3:'Very good'}
food_mapping={1:'Very sweet / Very salty / Junk food',2:'Healthy food'}
internet_mapping={2:'Have',1:'No'}
nursery_mapping={1:'Yes',0:'No'}
higheredu_mapping={1:'Yes',0:'No'}
transport_mapping={1:'Personal vehicle',0:'Other'}
extraclasses_mapping={2:'Yes',1:'No'}
gender_mapping={1:'Female',0:'Male'}
rev=lambda d:{v:k for k,v in d.items()}
family_income_rev=rev(family_income_mapping); edu_rev=rev(edu_mapping); relationship_rev=rev(relationship_mapping); food_rev=rev(food_mapping); internet_rev=rev(internet_mapping); nursery_rev=rev(nursery_mapping); higheredu_rev=rev(higheredu_mapping); transport_rev=rev(transport_mapping); extraclasses_rev=rev(extraclasses_mapping); gender_rev=rev(gender_mapping)

math_features=['පළවන වාරයේ ගණිතය ලකුණු','දෙවන වාරයේ ගණිතය ලකුණු ','6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?','13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?','5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?','3.සාමාන්\u200dය පවුලේ ආදායම','27.ඔබ නිතර ගන්නා ආහාර...?','8. දිනකට  නිවසේදී පාඩම් කිරීම සඳහා වැය කරන පැය සංඛ\u200dයාව','22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)','16.පවුලේ සබඳතා තත්ත්වය?','15.ඔබ  පෙර පාසල් ගොස් තිබේද?','14.ගෙදර අන්තර්ජාල පහසුකම් තිබේද?','2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී','17.ඔබ උසස් අධ්\u200dයාපනය ලබා ගැනීමට කැමතිද?']
science_features=['පළවන වාරයේ විද්\u200dයා ලකුණු','දෙවන වාරයේ විද්\u200dයා ලකුණු','6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?','13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?','27.ඔබ නිතර ගන්නා ආහාර...?','22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)','3.සාමාන්\u200dය පවුලේ ආදායම','5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?','16.පවුලේ සබඳතා තත්ත්වය?','2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී']

def align(df, model, expected):
    cols=getattr(model,'feature_names_in_',None) if model is not None else None
    cols=list(cols) if cols is not None else expected
    missing=[c for c in cols if c not in df.columns]
    if missing: raise ValueError('Missing model features: '+', '.join(missing))
    return df[cols]

def status():
    with st.expander('🔧 Model status'):
        st.write('Maths model:', '✅ Loaded' if math_model is not None else f'❌ {math_error}')
        st.write('Science model:', '✅ Loaded' if science_model is not None else f'❌ {science_error}')
        st.write('Science scaler:', '✅ Loaded' if science_scaler is not None else f'❌ {scaler_error}')

st.markdown('<div class="main-title">📚 Student Marks Predictor</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict 3rd-term Maths and Science marks using the trained machine-learning models.</div>',unsafe_allow_html=True)
status()
math_tab, science_tab = st.tabs(['📘 Maths Prediction','🔬 Science Prediction'])

with math_tab:
    st.header('📘 Maths 3rd-Term Marks Prediction')
    with st.form('math_form'):
        c1,c2=st.columns(2)
        with c1: math_term1=st.number_input('1st Term Maths Marks',0.0,200.0,0.0,1.0)
        with c2: math_term2=st.number_input('2nd Term Maths Marks',0.0,200.0,0.0,1.0)
        study_hours=st.number_input('Study Hours Per Day',0.0,24.0,1.0,0.5)
        c1,c2=st.columns(2)
        with c1:
            income=st.selectbox('Family Income',list(family_income_mapping.values()))
            mother=st.selectbox("Mother's Highest Education",list(edu_mapping.values()))
            father=st.selectbox("Father's Highest Education",list(edu_mapping.values()))
            extra=st.selectbox('Attend Extra Classes?',list(extraclasses_mapping.values()),index=1)
            food=st.selectbox('Food Usually Consumed',list(food_mapping.values()))
        with c2:
            transport=st.selectbox('Transport to School',list(transport_mapping.values()))
            relation=st.selectbox('Family Relationship',list(relationship_mapping.values()),index=1)
            nursery=st.selectbox('Attended Preschool?',list(nursery_mapping.values()))
            internet=st.selectbox('Internet at Home?',list(internet_mapping.values()))
            gender=st.selectbox('Gender',list(gender_mapping.values()))
            higher=st.selectbox('Interested in Higher Education?',list(higheredu_mapping.values()))
        submit=st.form_submit_button('🔮 Predict Maths Marks',use_container_width=True)
    if submit:
        if math_model is None: st.error('Maths model could not be loaded. Check the model file and package versions.')
        else:
            try:
                x=pd.DataFrame([{'පළවන වාරයේ ගණිතය ලකුණු':math_term1,'දෙවන වාරයේ ගණිතය ලකුණු ':math_term2,'6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?':edu_rev[mother],'13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?':extraclasses_rev[extra],'5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?':edu_rev[father],'3.සාමාන්\u200dය පවුලේ ආදායම':family_income_rev[income],'27.ඔබ නිතර ගන්නා ආහාර...?':food_rev[food],'8. දිනකට  නිවසේදී පාඩම් කිරීම සඳහා වැය කරන පැය සංඛ\u200dයාව':study_hours,'22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)':transport_rev[transport],'16.පවුලේ සබඳතා තත්ත්වය?':relationship_rev[relation],'15.ඔබ  පෙර පාසල් ගොස් තිබේද?':nursery_rev[nursery],'14.ගෙදර අන්තර්ජාල පහසුකම් තිබේද?':internet_rev[internet],'2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී':gender_rev[gender],'17.ඔබ උසස් අධ්\u200dයාපනය ලබා ගැනීමට කැමතිද?':higheredu_rev[higher]}])
                pred=float(math_model.predict(align(x,math_model,math_features))[0])
                st.success(f'📘 Predicted 3rd-Term Maths Marks: {pred:.2f}')
            except Exception as e: st.error(f'Maths prediction failed: {e}')

with science_tab:
    st.header('🔬 Science 3rd-Term Marks Prediction')
    with st.form('science_form'):
        c1,c2=st.columns(2)
        with c1: science_term1=st.number_input('1st Term Science Marks',0.0,200.0,0.0,1.0)
        with c2: science_term2=st.number_input('2nd Term Science Marks',0.0,200.0,0.0,1.0)
        c1,c2=st.columns(2)
        with c1:
            sincome=st.selectbox('Family Income',list(family_income_mapping.values()),key='s_income')
            smother=st.selectbox("Mother's Highest Education",list(edu_mapping.values()),key='s_mother')
            sfather=st.selectbox("Father's Highest Education",list(edu_mapping.values()),key='s_father')
            sextra=st.selectbox('Attend Extra Classes?',list(extraclasses_mapping.values()),index=1,key='s_extra')
        with c2:
            sfood=st.selectbox('Food Usually Consumed',list(food_mapping.values()),key='s_food')
            stransport=st.selectbox('Transport to School',list(transport_mapping.values()),key='s_transport')
            srelation=st.selectbox('Family Relationship',list(relationship_mapping.values()),index=1,key='s_relation')
            sgender=st.selectbox('Gender',list(gender_mapping.values()),key='s_gender')
        submit=st.form_submit_button('🔮 Predict Science Marks',use_container_width=True)
    if submit:
        if science_model is None: st.error('Science model could not be loaded. Check the model file and package versions.')
        elif science_scaler is None: st.error('Science scaler could not be loaded. Check gbr_science_scaler.pkl.')
        else:
            try:
                x=pd.DataFrame([{'ಪළවන වාරයේ විද්\u200dයා ලකුණු':science_term1,'දෙවන වාරයේ විද්\u200dයා ලකුණු':science_term2,'6. මවගේ \u200d ඉහලම අධ්\u200dයාපන සුදුසුකම?':edu_rev[smother],'13.ඔබ අමතර පන්ති වලට සහභාගී වෙන වද?':extraclasses_rev[sextra],'27.ඔබ නිතර ගන්නා ආහාර...?':food_rev[sfood],'22. පාසලට   ඔබ පැමිණෙන්නේ කෙසේද?_පුද්ගලික වාහන භාවිතයෙන්(බයිසිකලය,මෝටර් සයිකලය,ත්\u200dරීරෝද රථය,වෑන් රථයක........)':transport_rev[stransport],'3.සාමාන්\u200dය පවුලේ ආදායම':family_income_rev[sincome],'5.පියාගේ ඉහලම  අධ්\u200dයාපන සුදුසුකම?':edu_rev[sfather],'16.පවුලේ සබඳතා තත්ත්වය?':relationship_rev[srelation],'2.ස්ත්\u200dරී පුරුෂ භාවය_ස්ත්\u200dරී':gender_rev[sgender]}])
                x=align(x,science_model,science_features)
                pred=float(science_model.predict(science_scaler.transform(x))[0])
                st.success(f'🔬 Predicted 3rd-Term Science Marks: {pred:.2f}')
            except Exception as e: st.error(f'Science prediction failed: {e}')

st.markdown('---')
st.caption('📚 Student Marks Predictor | Machine Learning Research Project')
