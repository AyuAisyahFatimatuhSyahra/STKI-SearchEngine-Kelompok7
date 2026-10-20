"""Daftar stopword Bahasa Indonesia & Bahasa Inggris (tanpa unduhan eksternal)."""

STOPWORDS_ID = set("""
ada adalah adanya agar akan akhirnya aku akulah amat amatlah anda andalah antar
antara antaranya apa apaan apabila apakah apalagi apatah atau ataukah ataupun
bagai bagaikan bagaimana bagaimanakah bagaimanapun bagi bahkan bahwa bahwasanya
banyak beberapa begini beginian beginikah beginilah begitu begitukah begitulah
begitupun belum belumlah berapa berapakah berapalah berapapun bermacam bersama
betulkah biasa biasanya bila bilakah bisa bisakah boleh bolehkah bolehlah buat
bukan bukankah bukanlah bukannya cuma dahulu dalam dan dapat dari daripada dekat
demi demikian demikianlah dengan depan di dia dialah diantara diantaranya
dikarenakan dini diri dirinya disini disinilah dong dulu enggak enggaknya entah
entahlah hal hampir hanya hanyalah harus haruslah harusnya hendak hendaklah
hendaknya hingga ia ialah ibarat ingin inginkah inginkan ini inikah inilah itu
itukah itulah jadi jangan jangankan janganlah jika jikalau juga justru kala
kalau kalaulah kalaupun kalian kami kamilah kamu kamulah kan kapan kapankah
kapanpun karena karenanya ke kecil kemudian kenapa kepada kepadanya ketika
khususnya kini kinilah kiranya kita kitalah kok lagi lagian lah lain lainnya
lalu lama lamanya lebih macam maka makanya makin malah malahan mampu mampukah
mana manakala manalagi masih masihkah masing mau maupun melainkan melalui memang
mengapa mereka merekalah merupakan meski meskipun mungkin mungkinkah nah namun
nanti nantinya nyaris oleh olehnya pada padahal padanya paling pantas para pasti
pastilah per percuma pernah pula pun rupanya saat saatnya saja sajalah saling
sama sambil sampai sana sangat sangatlah saya sayalah se sebab sebabnya sebagai
sebagaimana sebagainya sebaliknya sebanyak sebegini sebegitu sebelum sebelumnya
sebenarnya seberapa sebetulnya sebisanya sebuah sedang sedangkan sedemikian
sedikit sedikitnya segala segalanya segera seharusnya sehingga sejak sejenak
sekali sekalian sekaligus sekalipun sekarang seketika sekiranya sekitar
sekitarnya sela selagi selain selaku selalu selama selamanya seluruh seluruhnya
semacam semakin semasih semaunya sementara sempat semua semuanya semula sendiri
sendirinya seolah seorang sepanjang sepantasnya seperti sepertinya sering
seringnya serta serupa sesaat sesama sesegera sesekali seseorang sesuatu
sesuatunya sesudah sesudahnya setelah seterusnya setiap setidaknya sewaktu
siapa siapakah siapapun sini sinilah suatu sudah sudahkah sudahlah supaya tadi
tadinya tak tanpa tapi telah tentang tentu tentulah tentunya terdiri terhadap
terhadapnya terlalu terlebih tersebut tersebutlah tertentu tetapi tiap tidak
tidakkah tidaklah toh untuk waduh wah wahai walau walaupun wong yaitu yakni
yang juga oleh pula yg dgn dll dsb dst
""".split())

STOPWORDS_EN = set("""
a about above across after afterwards again against all almost alone along
already also although always am among amongst an and another any anyhow anyone
anything anyway anywhere are around as at be became because become becomes
becoming been before beforehand behind being below beside besides between beyond
both but by can cannot could did do does doing done down due during each eg
either else elsewhere enough etc even ever every everyone everything everywhere
except few for former formerly from further had has have having he hence her
here hereafter hereby herein hers herself him himself his how however i ie if in
inc indeed into is it its itself just last latter latterly least less ltd made
many may me meanwhile might mine more moreover most mostly much must my myself
namely neither never nevertheless next no nobody none noone nor not nothing now
nowhere of off often on once one only onto or other others otherwise our ours
ourselves out over own per perhaps please rather re same seem seemed seeming
seems several she should since so some somehow someone something sometime
sometimes somewhere still such than that the their theirs them themselves then
thence there thereafter thereby therefore therein thereupon these they this
those though through throughout thru thus to together too toward towards under
until up upon us very via was we well were what whatever when whence whenever
where whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves s t can will just don should now et al
""".split())

STOPWORDS = STOPWORDS_ID | STOPWORDS_EN
