# Python 3.8
import smtplib
import time
import imaplib
import email
import traceback
# -------------------------------------------------
#
# Utility to read email from Gmail Using Python
#
# ------------------------------------------------
import numpy as np

import pickle

loaded_model = pickle.load(open("finalized_model.sav", "rb"))


SMTP_SERVER = "imap.gmail.com"
SMTP_PORT = 993

def read_email_from_gmail(FROM_EMAIL, FROM_PWD):
    try:
        mail = imaplib.IMAP4_SSL(SMTP_SERVER) #connection
        mail.login(FROM_EMAIL,FROM_PWD)
        mail.select('inbox')

        data = mail.search(None, 'UNSEEN')
        mail_ids = data[1]
        id_list = mail_ids[0].split()
        # print("LIST:")
        # print(id_list)
        first_email_id = int(id_list[0])
        # print("FIRST : ")
        # print(first_email_id)
        latest_email_id = int(id_list[-1])
        # print('\n')
        # print("LATEST :")
        # print(latest_email_id)

        new_data = []

        for i in range(latest_email_id,first_email_id-1, -1):
            data = mail.fetch(str(i), '(RFC822)' )
            for response_part in data:
                arr = response_part[0]
                if isinstance(arr, tuple):
                    msg = email.message_from_string(str(arr[1],'utf-8'))
                    email_subject = msg['subject']
                    email_from = msg['from']

                    print('From : ' + email_from)
                    print('Subject : ' + email_subject)

                    data = str(msg)

                    # Handling errors related to unicodenecode
                    try:
                        indexstart = data.find("ltr")
                        data2 = data[indexstart + 5: len(data)]
                        # print("DATA2")
                        # print(data2)
                        indexend1 = data2.find("<div>")
                        indexend2 = data2.find("</div>")
                        # print("INDEXEND")
                        # print(indexend1)
                        # print('\n')
                        # print(indexend2)

                        if(indexend1 == -1):
                            indexend1 = indexend2
                        if(indexend2 == -1):
                            indexend2 = indexend1


                        # printtng the required content which we need
                        # to extract from our email i.e our body
                        indexend = max(indexend2, indexend1)
                        # print("Message :" + data2[0: indexend] + '\n')
                        final_msg = data2[0: indexend]


                        # initializing sub list
                        sub_list = ["=C2=A0"," =C2=A0","=C2=A0 ", "<div>", "</div>", "<br>", "</= div>"]


                        # Remove substring list from String
                        # Using loop + replace()
                        for sub in sub_list:
                            final_msg = final_msg.replace(sub, ' ')

                        # printing result
                        print(final_msg)

                        l = []
                        l.append(final_msg)

                        result_pred = loaded_model.best_estimator_.predict(np.array(l))
                        print(result_pred)

                        new_data.append([email_from, final_msg, result_pred])


                    except UnicodeEncodeError as e:
                        pass

        return new_data


    except Exception as e:
        traceback.print_exc()
        print("here")
        print(str(e)) #list index out of range

