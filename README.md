# Image Retrieval system of Traditional Chinese Painting Based on Convolutional Neural Network

#ELG5378_IP_Project
### Image Processing and Image Communication

Traditional Chinese painting can be traced back nearly five thousand years and has created a large number of stunning and unique art forms. With the development of digital technology, digital art is considered one of the most convenient and effective ways to help educate and enhance the public's aesthetic taste. However, due to the long history of Chinese painting, each artist in different periods and dynasties has established various preferences and techniques for their own works. This has created great difficulty for the public to understand the various styles of art works and the connection between artists and art history periods. The expected use case of this application is to provide a search engine for similar art pieces to visitors, students, or art curators who are interested in learning about the history and related information behind the artwork while touring an exhibition hall or studying art in a university setting, but due to certain reasons, lacking direct access to name or author information about the artwork. They can use this application to obtain information about the history and related information of art works and similar pieces, but only when they are unable to directly obtain the name or author information for the artwork.

In our project：
1. we developed an automatic analysis system to analyze subtle differences between different art works and output the genre and artist. 
2. we considered data augmentation for oversampling, and applied the VGG-16 architecture to extract features from input images,
3. and compared the extracted features in a feature vector database. 
4. we establish the system based on Django fronted framework.

Finally, we presented the two highest similarity results.
