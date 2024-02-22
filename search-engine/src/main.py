import pandas as pd
from crawlers.vanguard import VanguardInsights
from crawlers.goldmansachs import GoldmanSachsInsights
from utils import DataBaseClass



if __name__ == "__main__":
    
    # Vanguard
    vanguard_insight_obj = VanguardInsights('https://advisors.vanguard.com/insights/all')
    vanguard_insight_list = vanguard_insight_obj.get_article_insight()
    vanguard_df = pd.DataFrame(vanguard_insight_list)

    #GoldmanSachs
    obj = GoldmanSachsInsights()
    goldman_df = obj.get_df_insights()
    

    # Store data in db

    obj_data_base_class = DataBaseClass('guest', 'Aa12345', 'localhost', 5432, 'insights_db')
    obj_data_base_class.store_data(vanguard_df, sql_table_name='insights_data')
    obj_data_base_class.store_data(goldman_df, sql_table_name='insights_data')

    