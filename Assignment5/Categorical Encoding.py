import numpy as np

# Categorical variables
colors = np.array(["Red", "Green", "Blue", "Green", "Red", "Blue"])
sizes = np.array(["Small", "Medium", "Large", "Small", "Large", "Medium"])
brands = np.array(["Nike", "Adidas", "Puma", "Nike", "Puma", "Adidas"])

#---------------------------------------------------------

#1 Label Encoding

# #과제에서 지정된 index 설정
# custom_order = ["Nike", "Adidas", "Puma"]

# #custom_order 순서대로 index mapping
# brand_to_index = {brand: idx for idx, brand in enumerate(custom_order)}

# #loop을 사용하여 brands 배열을 순회하며 각 브랜드를 index로 치환
# label_encoded_brand = np.array([brand_to_index[b] for b in brands])

unique_brands = np.unique(brands)  #['Adidas', 'Nike', 'Puma']
#unique_brands의 순서로 index하여 mapping dictionary 생성
brand_to_index = {brand: idx for idx, brand in enumerate(unique_brands)}
#loop을 이용하여 brands 배열 순회하며 각 브랜드를 index로 변환
label_encoded_brand = np.array([brand_to_index[b] for b in brands])

print("Original brands:", brands)
print("Encoded brands :", label_encoded_brand)

#2 Ordinal Encoding
#과제에서 지정된 mapping dictionary를 생성
size_mapping = {
    "Small": 1,
    "Medium": 2,
    "Large": 3
}

#mapping dictionary를 사용하여 사이즈들을 숫자로 변환
ordinal_encoded_sizes = np.array([size_mapping[size] for size in sizes])

print("Original sizes:", sizes)
print("Ordinal encoded sizes:", ordinal_encoded_sizes)

#3 One-Hot Encoding
#np.unique() 사용하여 unique한 feature 추출
unique_colors = np.unique(colors)

#배열 초기화
one_hot_encoded_color = np.zeros((colors.shape[0], unique_colors.shape[0]), dtype=int)

#반복문을 돌면서 각 색상의 인덱스 찾아서 1로 설정
#현재 색상의 고유 색상 리스트에서 index를 찾아(color_index) 1로 반환
#해당 위치[0 0 0]에 1을 넣어 encoding을 실행
for i, color in enumerate(colors):
    color_index = np.where(unique_colors == color)[0][0]
    one_hot_encoded_color[i, color_index] = 1

print("Unique colors:", unique_colors)
print("One-hot encoded array:\n", one_hot_encoded_color)

#앞선 encoded 데이터를 hstack(수평 통합)을 사용하여 새로운 feature matrix 생성
#Final shape should be 6x(3 + 1 + 1) = 6x5
final_feature_matrix = np.hstack([one_hot_encoded_color, ordinal_encoded_sizes.reshape(-1, 1), label_encoded_brand.reshape(-1, 1)])


print("Final feature matrix shape:", final_feature_matrix.shape)
print(final_feature_matrix)

#Short Reflection Questions (in comments)

#Why is one-hot encoding better for colors than label encoding?
#label encoding은 모델이 각 인덱스를 순서, 크기 등의 상하관계로 오해할 수 있다
#때문에 이런 관계가 존재하지 않은 feature(color)의 경우 one-hot encoding이 더 적합하다

#Why is ordinal encoding okay for sizes?
#사이즈는 상하관계가 존재한다(small<medium<large)
#이 관계를 숫자로 반영할 수 있는 ordinal encoding이 적합하다