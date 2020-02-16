#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <set>
#include <pthread.h>
#include <vector>
#include <map>

//#include <io.h>
//#include <direct.h>
//#include <errno.h>

using namespace std;

const float pi = 3.141592653589793238462643383;

int transeThreads = 8;
int transeTrainTimes = 1000;
int nbatches = 1;
int dimension = 100;
float transeAlpha = 0.001;
float margin = 1;  // former 1
int L1_flag = 1;
double combination_threshold = 3;
int combination_restriction = 5000;

string inPath = "./data/";
string outPath = "./res/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;
set<int> commonEntities;
vector<int> entitiesInKg1, entitiesInKg2;
map<int, int> correspondingEntity;
vector<float> combinationProbability;

struct Triple {
	int h, r, t;
};

Triple *trainHead, *trainTail, *trainList;

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

/*
	There are some math functions for the program initialization.
*/
unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float sigmoid(float x){
    return 1.0/(1.0 + exp(-x));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(float * con) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
//    for (int ii=0; ii < dimension; ii++)
//        *(con + ii) /= x;
//    float tp = 0.0;
//    for (int ii=0; ii < dimension; ii++)
//        tp += (*(con + ii)) * (*(con + ii));
//    float tr = 0.0;
//    tr = tp;
}

/*
	Read triples from the training file.
*/

int relationTotal, entityTotal, tripleTotal;
float *relationVec, *entityVec;
float *contextVec;  //add
float *relationVecDao, *entityVecDao;

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

//	tmp = 1;
//	relationTotal = 1563;

	relationVec = (float *)calloc(relationTotal * dimension, sizeof(float));
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			relationVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
//        norm(relationVec+i*dimension);  //ADD
	}

	fin = fopen((inPath + "newentity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

//	tmp = 1;
//	entityTotal = 20867;

	entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii<dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
//		norm(entityVec+i*dimension);
	}

	//add
    contextVec = (float *)calloc(dimension, sizeof(float));
    for (int i = 0; i < dimension; i++) {
        contextVec[i] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
    }
//    norm(contextVec);

	//initialize combinationProbability
	combinationProbability.resize(entityTotal);
	fill(combinationProbability.begin(), combinationProbability.end(), 0);

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	tripleTotal = 0;
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);
		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;
		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;
		tripleTotal++;
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(rigHead));
	memset(rigTail, -1, sizeof(rigTail));
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			//printf("hello");
			//printf("%d \n", tripleTotal);
			//printf("%d \n", entityTotal);
			//printf("%d \n", trainTail[i - 1].t);
			rigTail[trainTail[i - 1].t] = i - 1;
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	relationVecDao = (float*)calloc(dimension * relationTotal, sizeof(float));
	entityVecDao = (float*)calloc(dimension * entityTotal, sizeof(float));

	int commonTotal;
	fin = fopen((inPath + "common_entities2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &commonTotal);
	for(int i = 0;i<commonTotal;i++){
		int entId;
		tmp = fscanf(fin, "%d", &entId);
		commonEntities.insert(entId);
	}
    fclose(fin);

	FILE *fin_b;
	fin_b = fopen((inPath + "bid.txt").c_str(), "r");
	tmp = fscanf(fin_b, "%d", &commonTotal);
	for(int i = 0;i<commonTotal;i++){
		int entId;
		tmp = fscanf(fin_b, "%d", &entId);
		entitiesInKg1.push_back(entId);
	}
	fclose(fin_b);

	FILE *fin_h;
	fin_h = fopen((inPath + "hid.txt").c_str(), "r");
	tmp = fscanf(fin_h, "%d", &commonTotal);
	for(int i = 0;i<commonTotal;i++){
		int entId;
		tmp = fscanf(fin_h, "%d", &entId);
		entitiesInKg2.push_back(entId);
	}
	fclose(fin_h);

//	printf("%d known entities pairs.\n", commonTotal);
//	for(int i = 0;i<entityTotal;i++){
//		if(!commonEntities.count(i)){
//			if(i < 2563){
//				entitiesInKg1.push_back(i);
//			}
//			else entitiesInKg2.push_back(i);
//		}
//	}

}

/*
	Training process of transE.
*/

int transeLen;
int transeBatch;
float res;

float calc_sum(int e1, int e2, int rel) {
	float sum=0;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimension;

	//add
    float context_weight = 0.0;
    float temp = 0.0;
    temp = entityVec[last1 + 0];
    for (int i = 0; i < dimension; i++) {
        context_weight += relationVec[lastr + i]*contextVec[i];
    }

	for (int ii=0; ii < dimension; ii++) {
	    //importance bad disrupt
//	    sum += fabs(relationVec[lastr + ii] + entityVec[last2 + ii] - entityVec[last1 + ii]);
	    sum += fabs(relationVec[lastr + ii] + entityVec[last2 + ii] - context_weight*entityVec[last1 + ii]);
//	    sum += fabs(entityVec[last2 + ii] - entityVec[last1 + ii] - relationVec[lastr + ii]);
//	    sum += fabs(entityVec[last1 + ii] + relationVec[lastr + ii] - entityVec[last2 + ii]);
	}
	return sum;
}

void gradient1(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
    int lasta1 = e1_a * dimension;
    int lasta2 = e2_a * dimension;
    int lastar = rel_a * dimension;
    int lastb1 = e1_b * dimension;
    int lastb2 = e2_b * dimension;
    int lastbr = rel_b * dimension;
    for (int ii=0; ii  < dimension; ii++) {
        float x;
        x = (entityVec[lasta2 + ii]  + relationVec[lastar + ii] - entityVec[lasta1 + ii]);
        if (x > 0)
            x = transeAlpha;
        else
            x = -transeAlpha;
        relationVec[lastar + ii] -= x;
        entityVec[lasta1 + ii] += x;
        entityVec[lasta2 + ii] -= x;
        x = (entityVec[lastb2 + ii] + relationVec[lastbr + ii] - entityVec[lastb1 + ii]);
        if (x > 0)
            x = -transeAlpha;
        else
            x = transeAlpha;
        relationVec[lastbr + ii] -=  x;
        entityVec[lastb1 + ii] += x;
        entityVec[lastb2 + ii] -= x;
    }
}


void gradient(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimension;
	int lastb1 = e1_b * dimension;
	int lastb2 = e2_b * dimension;
	int lastbr = rel_b * dimension;

    //add
    float h_head = 0.0;
    float h_relation = 0.0;
    float h_context = 0.0;
    float h_context_weight = 0.0;

    for (int i = 0; i < dimension; i++) {
        h_head += relationVec[lastar + i] * contextVec[i];
    }
    for (int i = 0; i < dimension; i++) {
        h_relation += entityVec[lasta1 + i];
    }
    for (int i = 0; i < dimension; i++) {
        h_context += entityVec[lasta1 + i];
    }
    for (int i = 0; i < dimension; i++) {
        h_context_weight += relationVec[lastar + i] * contextVec[i];
    }

    float t_head = 0.0;
    float t_relation = 0.0;
    float t_context = 0.0;
    float t_context_weight = 0.0;
    for (int i = 0; i < dimension; i++) {
        t_head += relationVec[lastbr + i] * contextVec[i];
    }
    for (int i = 0; i < dimension; i++) {
        t_relation += entityVec[lastb1 + i];
    }
    for (int i = 0; i < dimension; i++) {
        t_context += entityVec[lastb1 + i];
    }
    for (int i = 0; i < dimension; i++) {
        t_context_weight += relationVec[lastbr + i] * contextVec[i];
    }

	for (int ii=0; ii  < dimension; ii++) {
		float x;
		x = (relationVec[lastar + ii] + entityVec[lasta2 + ii] -h_context_weight*entityVec[lasta1 + ii]);
//		x = (relationVec[lastar + ii] + entityVec[lasta2 + ii] -entityVec[lasta1 + ii]);
//		x = (relationVec[lastar + ii] + entityVec[lasta1 + ii] -entityVec[lasta2 + ii]);
//		x = (relationVec[lastar + ii] + entityVec[lasta2 + ii] -entityVec[lasta1 + ii]);
//		x = (entityVec[lasta2 + ii] - entityVec[lasta1 + ii] - relationVec[lastar + ii]);  //former
		if (x > 0)
			x = transeAlpha;
//			x = -transeAlpha;  //former
		else
			x = -transeAlpha;
//			x = transeAlpha;  //former

		//add r+t-context_weight*h
//		float context_temp = contextVec[ii]; // record contextVec value before gradient
//		float entity_value_b1 = entityVec[lastb1 + ii];  // position only here

        // get value before gradient
        float h_head_value = 0.0;
        float h_relation_value = relationVec[lastar+ii];
        float h_tail_value = 0.0;
        float h_context_value = contextVec[ii];

        float t_head_value = 0.0;
        float t_relation_value = relationVec[lastbr+ii];
        float t_tail_value = 0.0;
        float t_context_value = contextVec[ii];

		entityVec[lasta1 + ii] += h_head*x;
        relationVec[lastar + ii] -= x;
        relationVec[lastar + ii] += h_context_value*h_relation*x;
        entityVec[lasta2 + ii] -= x;
        contextVec[ii] += h_relation_value*h_context*x;

        //add r+t-h
//        relationVec[lastar + ii] -= x;  //r+t-h
//        entityVec[lasta1 + ii] += x;  //r+t-h
//        entityVec[lasta2 + ii] -= x;  //r+t-h

        // former
//        relationVec[lastar + ii] -= x;  //former
//		entityVec[lasta1 + ii] -= x;  //former
//		entityVec[lasta2 + ii] += x;  //former

//		x = (relationVec[lastbr + ii] + entityVec[lastb2 + ii] - entityVec[lastb1 + ii]);  //r+t-h
//		x = (relationVec[lastbr + ii] + entityVec[lastb1 + ii] - entityVec[lastb2 + ii]);  //former
//		x = (relationVec[lastbr + ii] + entityVec[lastb2 + ii] - entityVec[lastb1 + ii]);
		x = (relationVec[lastbr + ii] + entityVec[lastb2 + ii] - t_context_weight*entityVec[lastb1 + ii]);  //r+t-context_weight*h
		if (x > 0)
			x = -transeAlpha;
		else
			x = transeAlpha;
//        relationVec[lastbr + ii] -=  x;  //r+t-h
//        entityVec[lastb1 + ii] +=  x;  //r+t-h
//        entityVec[lastb2 + ii] -=  x;  //r+t-h

//		relationVec[lastbr + ii] -=  x;  //former
//        entityVec[lastb1 + ii] -=  x;  //former
//        entityVec[lastb2 + ii] +=  x;  //former

        //r+t-context_weight*h
        entityVec[lastb1 + ii] += t_head*x;
        relationVec[lastbr + ii] -= x;
        relationVec[lastbr + ii] += t_context_value*t_relation*x;
        entityVec[lasta2 + ii] -= x;
        contextVec[ii] += t_relation_value*t_context*x;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a);
	float sum2 = calc_sum(e1_b, e2_b, rel_b);
	float result = 0.0;
	result = sum1 - sum2;
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
//		printf("hello:\t");
//		printf("%.1f \n", res);
		gradient(e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transetrainMode(void *con) {
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = transeBatch / transeThreads; k >= 0; k--) {
		int j;
		int i = rand_max(id, transeLen);
		int pr = 500;
		int h1, t1, h2, t2,r;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
			h1 = trainList[i].h, t1 = trainList[i].t, r = trainList[i].r;
			h2 = trainList[i].h, t2 = j;
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
			h1 = trainList[i].h, t1 = trainList[i].t, r = trainList[i].r;
			h2 = j, t2 = trainList[i].t;
		}
		norm(relationVec + dimension * trainList[i].r);
		norm(entityVec + dimension * trainList[i].h);
		norm(entityVec + dimension * trainList[i].t);
		norm(entityVec + dimension * j);
		norm(contextVec);  // add
		if(float(randd(id)%1000)/1000.0 < combinationProbability[h1]){
		    int h1_cor = correspondingEntity[h1];
		    train_kb(h1_cor, t1, r, h2, t2, r);
		    norm(entityVec + dimension * h1_cor);
		}
		if(float(randd(id)%1000)/1000.0 < combinationProbability[h2]){
		    int h2_cor = correspondingEntity[h2];
		    train_kb(h1, t1, r, h2_cor, t2, r);
		    norm(entityVec + dimension * h2_cor);
		}
		if(float(randd(id)%1000)/1000.0 < combinationProbability[t1]){
		    int t1_cor = correspondingEntity[t1];
		    train_kb(h1, t1_cor, r, h2, t2, r);
		    norm(entityVec + dimension * t1_cor);
		}
		if(float(randd(id)%1000)/1000.0 < combinationProbability[t2]){
		    int t2_cor = correspondingEntity[t2];
		    train_kb(h1, t1, r, h2, t2_cor, r);
		    norm(entityVec + dimension * t2_cor);
		}
		norm(relationVec + dimension * trainList[i].r);
		norm(entityVec + dimension * trainList[i].h);
		norm(entityVec + dimension * trainList[i].t);
		norm(entityVec + dimension * j);
        norm(contextVec);  // add
	}
}

double calc_distance(int ent1, int ent2){
    double sum=0;
    if (L1_flag)
        for (int ii=0; ii<dimension; ii++)
            sum+=fabs(entityVec[ent1*dimension + ii]-entityVec[ent2*dimension + ii]);
    else
        for (int ii=0; ii<dimension; ii++)
            sum+=pow(entityVec[ent1*dimension + ii]-entityVec[ent2*dimension + ii], 2);
    return sum;
}

void do_combine(){
	time_t beginTimer, endTimer;
	time(&beginTimer);
	printf("Combination begins.\n");
	vector<pair<double, pair<int, int> > > distance2entitiesPair;
	for(auto &i : entitiesInKg1)
		for(auto &j : entitiesInKg2)
			distance2entitiesPair.push_back(make_pair(calc_distance(i, j), make_pair(i, j)));
    sort(distance2entitiesPair.begin(), distance2entitiesPair.end());
    set<int> occupied;
    float minimalDistance = 0;
    for(auto &i : distance2entitiesPair){
		if(i.first > 0){
	    	minimalDistance = i.first;
	    	break;
		}
    }
    printf("Minimal distance is %lf \n", minimalDistance);
	correspondingEntity.clear();
	fill(combinationProbability.begin(), combinationProbability.end(), 0);
	int combination_counter = 0;
	for(auto &i: distance2entitiesPair){
		int dis = i.first, ent1 = i.second.first, ent2 = i.second.second;
		if(dis > combination_threshold) break;
		if(occupied.count(ent1) || occupied.count(ent2)) continue;
		correspondingEntity[ent1] = ent2;
		correspondingEntity[ent2] = ent1;
		printf("Combined %d and %d\n", ent1, ent2);
		occupied.insert(ent1);
		occupied.insert(ent2);
		combinationProbability[ent1] = sigmoid(combination_threshold - dis);
		combinationProbability[ent2] = sigmoid(combination_threshold - dis);
		if(combination_counter == combination_restriction) break;
		combination_counter++;
    }
	time(&endTimer);
	printf("Using %.f seconds to combine %d entities pairs.\n", difftime(endTimer, beginTimer), combination_counter);
	combination_restriction += 1000;
}
void out_transe(string);
void* train_transe(void *con) {
	transeLen = tripleTotal;
	transeBatch = transeLen / nbatches;
	next_random = (unsigned long long *)calloc(transeThreads, sizeof(unsigned long long));
	for (int epoch = 0; epoch < transeTrainTimes; epoch++) {
//		if(epoch > 999 && epoch % 500 == 0){
		if(epoch > 999 && epoch % 6000 == 0){
			do_combine();
		}
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transeThreads * sizeof(pthread_t));
			for (int a = 0; a < transeThreads; a++)
				pthread_create(&pt[a], NULL, transetrainMode,  (void*)a);
			for (int a = 0; a < transeThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
		}
		printf("epoch %d %f\n", epoch, res);
		fflush(stdout);
	}
}

/*
	Get the results of transE.
*/

void out_transe(string iter = "") {
		FILE* f2 = fopen((outPath + "relation2vec" + iter + ".bern_my2").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec" + iter + ".bern_my2").c_str(), "w");
		for (int i=0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
}

/*
	Main function
*/

int main() {
	init();
	train_transe(NULL);
	out_transe();
	return 0;
}
