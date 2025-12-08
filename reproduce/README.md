以下會放個階段應當用到的值
Offline Indexing 階段 (建置知識庫/準備課本) =>
_id: 當索引
context: 切成 chunking => 請LLM做NER跟OpenIE => 給Nvdia轉成embedding

offline程式順序
1.原始文本段落
=> 會被轉成 chunks
2.LLM 找出 NER example: "Bill Gates", "Microsoft"。 
=> 會被轉成 Entity Embeddings
3.LLM 找出 OpenIE(triple) example: LLM 生成結構化的三元組 (Bill Gates, founder of, Microsoft)。 
=> 會被轉成 Fact Embeddings

Online程式順序
1. 使用者查詢 (User Query) Example： "Who created Microsoft?"
=> 會被 NV-Embed 轉成 Query Embedding (特別是 query_to_fact 模式)。
2. 向量檢索 (Vector Search)
=> 拿 Query Embedding 去跟 Offline 建好的 Fact Embeddings 比對相似度。
3. LLM 過濾 (Recognition Memory) => 篩選出 Verified Facts (確認事實)。
4. 圖搜尋 (Graph Search - PPR)
=> 從這些確認的事實出發，透過 Entity (實體節點) 在知識圖譜上遊走。
5. 最終輸出 
=> 回傳這些 Chunks 的原始文本 給使用者或生成模型。




context,Offline,原材料 (用來蓋圖書館/知識圖譜)
question,Online,查詢指令 (使用者的問題)
type,Online,策略開關 (決定要單跳還是多跳搜尋)

answer,Training,最終目標 (用來改考卷)
supporting_facts,Training,檢索目標 (教檢索器該抓哪幾篇)
evidences,Training,推理路徑 (教模型該怎麼思考)

1. _id
意思：該數據樣本的唯一識別碼（Unique Identifier）。
用途：用於在數據集中追蹤、索引或區分不同的問題。
範例值："83bf3b5a0bd911eba7f7acde48001122"

2. type
意思：問題的推理類型。
用途：告訴模型解決這個問題需要哪種邏輯。
在此例中為 "compositional"（組合式/推理式），意味著需要將多個訊息片段拼湊起來（先找出 Lothair II 的母親是誰，再找出她什麼時候過世）。其他常見類型可能還有 "comparison"（比較題）。
範例值："compositional"

3. question
意思：需要回答的自然語言問題。
用途：模型的輸入，模型需要根據 context 來回答此問題。
範例值："When did Lothair Ii's mother die?"（洛泰爾二世的母親什麼時候去世？）

4. context
意思：提供給模型的參考文檔/知識庫。
結構：這是一個列表（List），其中的每個元素也是一個列表，包含兩個部分：
標題 (Title)：該段落的主題或文章名稱（例如 "Teutberga"）。
句子列表 (List of Sentences)：該段落的具體內容，被拆分成一句一句的字串。
用途：模型必須從這些雜亂的段落中檢索出相關資訊。
範例值：包含了 "Teutberga", "Lothair II", "Ermengarde of Tours" 等多個條目的詳細文本。

5. entity_ids
意思：問題中涉及的關鍵實體的 ID。
用途：通常對應到知識圖譜（如 Wikidata）的 ID。這裡將兩個相關實體連接起來了。
Q298945 是 Lothair II（洛泰爾二世）。
Q235653 是 Ermengarde of Tours（圖爾的艾敏加德，即他母親）。
範例值："Q298945_Q235653"

6. supporting_facts
意思：支撐事實（Gold Standard Evidence）。這是回答問題所必需引用的具體句子。
結構：[文章標題, 句子索引]。
用途：用於訓練模型不僅要給出答案，還要能解釋「為什麼」（可解釋性 AI）。
["Lothair II", 1]：引用 "Lothair II" 這篇文章的第 2 句（索引從 0 開始），該句提到他的母親是 "Ermengarde of Tours"。
["Ermengarde of Tours", 0]：引用 "Ermengarde of Tours" 這篇文章的第 1 句，該句提到她死於 "20 March 851"。
範例值：列出了上述兩個關鍵證據位置。

7. evidences
意思：推理鏈/證據鏈，以結構化的三元組（Subject, Relation, Object）形式呈現。
用途：展示邏輯推導的過程。
第一步：從 Lothair II 找出 mother 是 Ermengarde of Tours。
第二步：從 Ermengarde of Tours 找出 date of death 是 20 March 851。
範例值：包含了上述兩個邏輯步驟。

8. answer
意思：問題的標準答案。
用途：用於計算模型預測的準確率（如 EM 或 F1 分數）。
範例值："20 March 851"

9. evidences_id
意思：對應於 evidences 的 ID 版本。
用途：將文字實體替換為其對應的知識圖譜 ID（如 Wikidata ID），便於機器處理結構化數據。
Lothair II -> Q298945
Ermengarde of Tours -> Q235653
範例值：[["Q298945", "mother", "Q235653"], ...]

10. answer_id
意思：答案實體的 ID。
用途：如果答案是一個特定的實體（如「歐巴馬」），這裡會放他的 ID。
範例值：null。因為此題的答案是「日期」（20 March 851），這通常被視為數值或字串，而不是一個知識圖譜中的實體 ID，所以為空。